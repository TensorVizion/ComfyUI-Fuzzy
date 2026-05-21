import asyncio
import functools
import json
import logging
import os
import urllib.parse
import uuid
from typing import Any
from io import BytesIO
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher

from aiohttp import web
from pydantic import ValidationError
from PIL import Image

import folder_paths
from app import user_manager
from app.assets.api import schemas_in, schemas_out
from app.assets.services import schemas
from app.assets.api.schemas_in import (
    AssetValidationError,
    UploadError,
)
from app.assets.helpers import validate_blake3_hash
from app.assets.api.upload import (
    delete_temp_file_if_exists,
    parse_multipart_upload,
)
from app.assets.seeder import ScanInProgressError, asset_seeder
from app.assets.services import (
    DependencyMissingError,
    HashMismatchError,
    apply_tags,
    asset_exists,
    create_from_hash,
    delete_asset_reference,
    get_asset_detail,
    list_assets_page,
    list_tags,
    remove_tags,
    resolve_asset_for_download,
    update_asset_metadata,
    upload_from_temp_path,
)
from app.assets.services.tagging import list_tag_histogram

ROUTES = web.RouteTableDef()
USER_MANAGER: user_manager.UserManager | None = None
_ASSETS_ENABLED = False

# Chunked upload session tracking
_UPLOAD_SESSIONS = {}  # upload_id -> {chunks: {}, total_chunks: int, metadata: {}, created_at: datetime}

# Response cache for asset metadata
_METADATA_CACHE = {}  # asset_id -> {data: {}, timestamp: datetime}
_CACHE_TTL = 300  # 5 minutes

# Undo/redo history for asset operations
_OPERATION_HISTORY = {}  # user_id -> [{operation: str, asset_id: str, previous_state: dict, timestamp: datetime}]
_MAX_HISTORY_SIZE = 50


def _get_cached_metadata(asset_id: str) -> dict | None:
    """Get cached asset metadata if available and not expired."""
    if asset_id in _METADATA_CACHE:
        cache_entry = _METADATA_CACHE[asset_id]
        if (datetime.now() - cache_entry["timestamp"]).total_seconds() < _CACHE_TTL:
            return cache_entry["data"]
        else:
            del _METADATA_CACHE[asset_id]
    return None


def _set_cached_metadata(asset_id: str, data: dict) -> None:
    """Cache asset metadata with timestamp."""
    _METADATA_CACHE[asset_id] = {
        "data": data,
        "timestamp": datetime.now()
    }


def _invalidate_cache(asset_id: str) -> None:
    """Invalidate cache for a specific asset."""
    if asset_id in _METADATA_CACHE:
        del _METADATA_CACHE[asset_id]


def _fuzzy_match_score(query: str, text: str) -> float:
    """Calculate fuzzy match score between query and text."""
    if not query or not text:
        return 0.0
    return SequenceMatcher(None, query.lower(), text.lower()).ratio()


def _filter_by_fuzzy_search(assets: list, query: str, threshold: float = 0.6) -> list:
    """Filter assets by fuzzy search with threshold."""
    if not query:
        return assets
    return [
        asset for asset in assets
        if _fuzzy_match_score(query, asset.ref.name) >= threshold
    ]


def _record_operation(user_id: str, operation: str, asset_id: str, previous_state: dict) -> None:
    """Record an operation for undo/redo support."""
    if user_id not in _OPERATION_HISTORY:
        _OPERATION_HISTORY[user_id] = []
    
    _OPERATION_HISTORY[user_id].append({
        "operation": operation,
        "asset_id": asset_id,
        "previous_state": previous_state,
        "timestamp": datetime.now()
    })
    
    # Limit history size
    if len(_OPERATION_HISTORY[user_id]) > _MAX_HISTORY_SIZE:
        _OPERATION_HISTORY[user_id].pop(0)


def _get_last_operation(user_id: str) -> dict | None:
    """Get the last operation for a user."""
    if user_id in _OPERATION_HISTORY and _OPERATION_HISTORY[user_id]:
        return _OPERATION_HISTORY[user_id][-1]
    return None


def _undo_operation(user_id: str) -> dict | None:
    """Undo the last operation for a user."""
    if user_id in _OPERATION_HISTORY and _OPERATION_HISTORY[user_id]:
        return _OPERATION_HISTORY[user_id].pop()
    return None


def _require_assets_feature_enabled(handler):
    @functools.wraps(handler)
    async def wrapper(request: web.Request) -> web.Response:
        if not _ASSETS_ENABLED:
            return _build_error_response(
                503,
                "SERVICE_DISABLED",
                "Assets system is disabled. Start the server with --enable-assets to use this feature.",
            )
        return await handler(request)

    return wrapper


# UUID regex (canonical hyphenated form, case-insensitive)
UUID_RE = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"


def get_query_dict(request: web.Request) -> dict[str, Any]:
    """Gets a dictionary of query parameters from the request.

    request.query is a MultiMapping[str], needs to be converted to a dict
    to be validated by Pydantic.
    """
    query_dict = {
        key: request.query.getall(key)
        if len(request.query.getall(key)) > 1
        else request.query.get(key)
        for key in request.query.keys()
    }
    return query_dict


# Note to any custom node developers reading this code:
# The assets system is not yet fully implemented,
# do not rely on the code in /app/assets remaining the same.


def register_assets_routes(
    app: web.Application,
    user_manager_instance: user_manager.UserManager | None = None,
) -> None:
    global USER_MANAGER, _ASSETS_ENABLED
    if user_manager_instance is not None:
        USER_MANAGER = user_manager_instance
        _ASSETS_ENABLED = True
    app.add_routes(ROUTES)


def disable_assets_routes() -> None:
    """Disable asset routes at runtime (e.g. after DB init failure)."""
    global _ASSETS_ENABLED
    _ASSETS_ENABLED = False


ERROR_HELP_MAP = {
    "INVALID_HASH": "Hash must be in format 'blake3:<hex>'. Ensure you're using the correct hash format.",
    "ASSET_NOT_FOUND": "The requested asset could not be found. Check the asset ID and try again.",
    "FILE_NOT_FOUND": "The underlying file is missing from disk. It may have been moved or deleted.",
    "INVALID_BODY": "The request body is invalid. Check the required fields and data types.",
    "INVALID_QUERY": "The query parameters are invalid. Check the parameter names and values.",
    "INVALID_JSON": "The request body must be valid JSON. Ensure proper JSON formatting.",
    "SERVICE_DISABLED": "The assets system is disabled. Start the server with --enable-assets to use this feature.",
    "FORBIDDEN": "You don't have permission to perform this action on this asset.",
    "BACKEND_UNSUPPORTED": "This operation is not supported by the current backend configuration.",
    "HASH_MISMATCH": "The provided hash doesn't match the file content. The file may be corrupted.",
    "DEPENDENCY_MISSING": "Required dependencies are missing. Check the server logs for details.",
    "MISSING_INPUT": "No file was uploaded and the hash was not found. Provide either a file or a valid hash.",
    "INTERNAL": "An unexpected server error occurred. Please try again or contact support if the issue persists.",
    "UNSUPPORTED_MEDIA_TYPE": "This file type is not supported for thumbnail generation.",
}


# Thumbnail cache directory
THUMBNAIL_CACHE_DIR = Path(__file__).parents[3] / "thumbnails"
THUMBNAIL_CACHE_DIR.mkdir(exist_ok=True)


def _generate_thumbnail(image_path: str, size: tuple[int, int] = (256, 256)) -> BytesIO:
    """Generate a thumbnail for an image file."""
    try:
        with Image.open(image_path) as img:
            img.thumbnail(size, Image.Resampling.LANCZOS)
            thumb_io = BytesIO()
            img.save(thumb_io, format="JPEG", quality=85, optimize=True)
            thumb_io.seek(0)
            return thumb_io
    except Exception as e:
        logging.error(f"Failed to generate thumbnail for {image_path}: {e}")
        raise


def _build_error_response(
    status: int, code: str, message: str, details: dict | None = None
) -> web.Response:
    return web.json_response(
        {
            "error": {
                "code": code,
                "message": message,
                "details": details or {},
                "help": ERROR_HELP_MAP.get(code, ""),
            }
        },
        status=status,
    )


def _build_validation_error_response(code: str, ve: ValidationError) -> web.Response:
    errors = json.loads(ve.json())
    return _build_error_response(400, code, "Validation failed.", {"errors": errors})


def _validate_sort_field(requested: str | None) -> str:
    if not requested:
        return "created_at"
    v = requested.lower()
    if v in {"name", "created_at", "updated_at", "size", "last_access_time"}:
        return v
    return "created_at"


def _build_preview_url_from_view(tags: list[str], user_metadata: dict[str, Any] | None) -> str | None:
    """Build a /api/view preview URL from asset tags and user_metadata filename."""
    if not user_metadata:
        return None
    filename = user_metadata.get("filename")
    if not filename:
        return None

    if "input" in tags:
        view_type = "input"
    elif "output" in tags:
        view_type = "output"
    else:
        return None

    subfolder = ""
    if "/" in filename:
        subfolder, filename = filename.rsplit("/", 1)

    encoded_filename = urllib.parse.quote(filename, safe="")
    url = f"/api/view?type={view_type}&filename={encoded_filename}"
    if subfolder:
        url += f"&subfolder={urllib.parse.quote(subfolder, safe='')}"
    return url


def _build_asset_response(result: schemas.AssetDetailResult | schemas.UploadResult) -> schemas_out.Asset:
    """Build an Asset response from a service result."""
    if result.ref.preview_id:
        preview_detail = get_asset_detail(result.ref.preview_id)
        if preview_detail:
            preview_url = _build_preview_url_from_view(preview_detail.tags, preview_detail.ref.user_metadata)
        else:
            preview_url = None
    else:
        preview_url = _build_preview_url_from_view(result.tags, result.ref.user_metadata)
    
    # Build accessibility metadata
    accessibility_info = {
        "label": result.ref.name,
        "description": f"Asset of type {result.asset.mime_type if result.asset else 'unknown'}",
        "tags": result.tags,
        "size_human": _human_readable_size(int(result.asset.size_bytes)) if result.asset else None,
        "created_date": result.ref.created_at.strftime("%B %d, %Y") if result.ref.created_at else None,
    }
    
    return schemas_out.Asset(
        id=result.ref.id,
        name=result.ref.name,
        asset_hash=result.asset.hash if result.asset else None,
        size=int(result.asset.size_bytes) if result.asset else None,
        mime_type=result.asset.mime_type if result.asset else None,
        tags=result.tags,
        preview_url=preview_url,
        preview_id=result.ref.preview_id,
        user_metadata=result.ref.user_metadata or {},
        metadata=result.ref.system_metadata,
        job_id=result.ref.job_id,
        prompt_id=result.ref.job_id,  # deprecated: mirrors job_id for cloud compat
        created_at=result.ref.created_at,
        updated_at=result.ref.updated_at,
        last_access_time=result.ref.last_access_time,
        accessibility=accessibility_info,
    )


def _human_readable_size(size_bytes: int) -> str:
    """Convert bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


@ROUTES.head("/api/assets/hash/{hash}")
@_require_assets_feature_enabled
async def head_asset_by_hash(request: web.Request) -> web.Response:
    hash_str = request.match_info.get("hash", "").strip().lower()
    try:
        hash_str = validate_blake3_hash(hash_str)
    except ValueError:
        return _build_error_response(
            400, "INVALID_HASH", "hash must be like 'blake3:<hex>'"
        )
    exists = asset_exists(hash_str)
    return web.Response(status=200 if exists else 404)


@ROUTES.get("/api/assets")
@_require_assets_feature_enabled
async def list_assets_route(request: web.Request) -> web.Response:
    """
    GET request to list assets.
    """
    query_dict = get_query_dict(request)
    try:
        q = schemas_in.ListAssetsQuery.model_validate(query_dict)
    except ValidationError as ve:
        return _build_validation_error_response("INVALID_QUERY", ve)

    sort = _validate_sort_field(q.sort)
    order_candidate = (q.order or "desc").lower()
    order = order_candidate if order_candidate in {"asc", "desc"} else "desc"

    # Use cursor-based pagination if cursor is provided
    if q.cursor:
        try:
            cursor_data = json.loads(q.cursor)
            offset = cursor_data.get("offset", 0)
        except (json.JSONDecodeError, AttributeError):
            offset = q.offset
    else:
        offset = q.offset

    result = list_assets_page(
        owner_id=USER_MANAGER.get_request_user_id(request),
        include_tags=q.include_tags,
        exclude_tags=q.exclude_tags,
        name_contains=q.name_contains,
        metadata_filter=q.metadata_filter,
        limit=q.limit,
        offset=offset,
        sort=sort,
        order=order,
    )

    # Apply fuzzy search if provided
    filtered_items = result.items
    if q.fuzzy_search:
        filtered_items = _filter_by_fuzzy_search(result.items, q.fuzzy_search)
        # Sort by fuzzy match score
        filtered_items.sort(
            key=lambda x: _fuzzy_match_score(q.fuzzy_search, x.ref.name),
            reverse=True
        )

    summaries = [_build_asset_response(item) for item in filtered_items]
    
    # Generate next cursor if there are more results
    next_cursor = None
    has_more = (offset + len(summaries)) < result.total
    if has_more and summaries:
        next_cursor = json.dumps({"offset": offset + len(summaries), "sort": sort, "order": order})

    payload = schemas_out.AssetsList(
        assets=summaries,
        total=len(filtered_items) if q.fuzzy_search else result.total,
        has_more=has_more,
        next_cursor=next_cursor,
    )
    return web.json_response(payload.model_dump(mode="json", exclude_none=True))


@ROUTES.get(f"/api/assets/{{id:{UUID_RE}}}")
@_require_assets_feature_enabled
async def get_asset_route(request: web.Request) -> web.Response:
    """
    GET request to get an asset's info as JSON.
    """
    reference_id = str(uuid.UUID(request.match_info["id"]))
    
    # Check cache first
    cached_data = _get_cached_metadata(reference_id)
    if cached_data:
        return web.json_response(cached_data, status=200)
    
    try:
        result = get_asset_detail(
            reference_id=reference_id,
            owner_id=USER_MANAGER.get_request_user_id(request),
        )
        if not result:
            return _build_error_response(
                404,
                "ASSET_NOT_FOUND",
                f"AssetReference {reference_id} not found",
                {"id": reference_id},
            )

        payload = _build_asset_response(result)
        payload_dict = payload.model_dump(mode="json", exclude_none=True)
        
        # Cache the response
        _set_cached_metadata(reference_id, payload_dict)
        
    except ValueError as e:
        return _build_error_response(
            404, "ASSET_NOT_FOUND", str(e), {"id": reference_id}
        )
    except Exception:
        logging.exception(
            "get_asset failed for reference_id=%s, owner_id=%s",
            reference_id,
            USER_MANAGER.get_request_user_id(request),
        )
        return _build_error_response(500, "INTERNAL", "Unexpected server error.")
    return web.json_response(payload_dict, status=200)


@ROUTES.get(f"/api/assets/{{id:{UUID_RE}}}/thumbnail")
@_require_assets_feature_enabled
async def get_asset_thumbnail(request: web.Request) -> web.Response:
    """Get a thumbnail for an asset."""
    reference_id = str(uuid.UUID(request.match_info["id"]))
    size_param = request.query.get("size", "256")
    try:
        size = int(size_param)
        if size < 64 or size > 1024:
            size = 256
    except ValueError:
        size = 256

    try:
        result = resolve_asset_for_download(
            reference_id=reference_id,
            owner_id=USER_MANAGER.get_request_user_id(request),
        )
        abs_path = result.abs_path
        content_type = result.content_type
    except ValueError as ve:
        return _build_error_response(404, "ASSET_NOT_FOUND", str(ve))
    except NotImplementedError as nie:
        return _build_error_response(501, "BACKEND_UNSUPPORTED", str(nie))
    except FileNotFoundError:
        return _build_error_response(
            404, "FILE_NOT_FOUND", "Underlying file not found on disk."
        )

    # Check if file is an image
    if not content_type or not content_type.startswith("image/"):
        return _build_error_response(
            400, "UNSUPPORTED_MEDIA_TYPE", "Thumbnails are only available for image assets."
        )

    # Check cache
    cache_key = f"{reference_id}_{size}.jpg"
    cache_path = THUMBNAIL_CACHE_DIR / cache_key

    if cache_path.exists():
        return web.FileResponse(
            cache_path,
            content_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    # Generate thumbnail
    try:
        thumb_io = _generate_thumbnail(abs_path, (size, size))
        
        # Save to cache
        with open(cache_path, "wb") as f:
            f.write(thumb_io.getvalue())
        
        thumb_io.seek(0)
        return web.Response(
            body=thumb_io.getvalue(),
            content_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=86400"},
        )
    except Exception as e:
        logging.exception("Failed to generate thumbnail for asset %s", reference_id)
        return _build_error_response(500, "INTERNAL", "Failed to generate thumbnail")


@ROUTES.get(f"/api/assets/{{id:{UUID_RE}}}/content")
@_require_assets_feature_enabled
async def download_asset_content(request: web.Request) -> web.Response:
    disposition = request.query.get("disposition", "attachment").lower().strip()
    if disposition not in {"inline", "attachment"}:
        disposition = "attachment"

    try:
        result = resolve_asset_for_download(
            reference_id=str(uuid.UUID(request.match_info["id"])),
            owner_id=USER_MANAGER.get_request_user_id(request),
        )
        abs_path = result.abs_path
        content_type = result.content_type
        filename = result.download_name
    except ValueError as ve:
        return _build_error_response(404, "ASSET_NOT_FOUND", str(ve))
    except NotImplementedError as nie:
        return _build_error_response(501, "BACKEND_UNSUPPORTED", str(nie))
    except FileNotFoundError:
        return _build_error_response(
            404, "FILE_NOT_FOUND", "Underlying file not found on disk."
        )

    _DANGEROUS_MIME_TYPES = {
        "text/html", "text/html-sandboxed", "application/xhtml+xml",
        "text/javascript", "text/css",
    }
    if content_type in _DANGEROUS_MIME_TYPES:
        content_type = "application/octet-stream"

    safe_name = (filename or "").replace("\r", "").replace("\n", "")
    encoded = urllib.parse.quote(safe_name)
    cd = f"{disposition}; filename*=UTF-8''{encoded}"

    file_size = os.path.getsize(abs_path)
    size_mb = file_size / (1024 * 1024)
    logging.info(
        "download_asset_content: path=%s, size=%d bytes (%.2f MB), type=%s, name=%s",
        abs_path,
        file_size,
        size_mb,
        content_type,
        filename,
    )

    async def stream_file_chunks():
        chunk_size = 64 * 1024
        with open(abs_path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    return web.Response(
        body=stream_file_chunks(),
        content_type=content_type,
        headers={
            "Content-Disposition": cd,
            "Content-Length": str(file_size),
            "X-Content-Type-Options": "nosniff",
        },
    )


@ROUTES.post("/api/assets/from-hash")
@_require_assets_feature_enabled
async def create_asset_from_hash_route(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
        body = schemas_in.CreateFromHashBody.model_validate(payload)
    except ValidationError as ve:
        return _build_validation_error_response("INVALID_BODY", ve)
    except Exception:
        return _build_error_response(
            400, "INVALID_JSON", "Request body must be valid JSON."
        )

    # Derive name from hash if not provided
    name = body.name
    if name is None:
        name = body.hash.split(":", 1)[1] if ":" in body.hash else body.hash

    result = create_from_hash(
        hash_str=body.hash,
        name=name,
        tags=body.tags,
        user_metadata=body.user_metadata,
        owner_id=USER_MANAGER.get_request_user_id(request),
        mime_type=body.mime_type,
        preview_id=body.preview_id,
    )
    if result is None:
        return _build_error_response(
            404, "ASSET_NOT_FOUND", f"Asset content {body.hash} does not exist"
        )

    asset = _build_asset_response(result)
    payload_out = schemas_out.AssetCreated(
        **asset.model_dump(),
        created_new=result.created_new,
    )
    return web.json_response(payload_out.model_dump(mode="json", exclude_none=True), status=201)


@ROUTES.post("/api/assets")
@_require_assets_feature_enabled
async def upload_asset(request: web.Request) -> web.Response:
    """Multipart/form-data endpoint for Asset uploads with chunked upload support."""
    try:
        parsed = await parse_multipart_upload(request, check_hash_exists=asset_exists)
    except UploadError as e:
        return _build_error_response(e.status, e.code, e.message)

    owner_id = USER_MANAGER.get_request_user_id(request)

    try:
        spec = schemas_in.UploadAssetSpec.model_validate(
            {
                "tags": parsed.tags_raw,
                "name": parsed.provided_name,
                "user_metadata": parsed.user_metadata_raw,
                "hash": parsed.provided_hash,
                "mime_type": parsed.provided_mime_type,
                "preview_id": parsed.provided_preview_id,
                "chunk_number": None,  # Will be set from form data if present
                "total_chunks": None,
                "upload_id": None,
            }
        )
    except ValidationError as ve:
        delete_temp_file_if_exists(parsed.tmp_path)
        return _build_error_response(
            400, "INVALID_BODY", f"Validation failed: {ve.json()}"
        )

    # Handle chunked uploads
    if spec.upload_id and spec.chunk_number and spec.total_chunks:
        return await _handle_chunked_upload(request, parsed, spec, owner_id)

    if spec.tags and spec.tags[0] == "models":
        if (
            len(spec.tags) < 2
            or spec.tags[1] not in folder_paths.folder_names_and_paths
        ):
            delete_temp_file_if_exists(parsed.tmp_path)
            category = spec.tags[1] if len(spec.tags) >= 2 else ""
            return _build_error_response(
                400, "INVALID_BODY", f"unknown models category '{category}'"
            )

    try:
        # Fast path: hash exists, create AssetReference without writing anything
        if spec.hash and parsed.provided_hash_exists is True:
            result = create_from_hash(
                hash_str=spec.hash,
                name=spec.name or (spec.hash.split(":", 1)[1]),
                tags=spec.tags,
                user_metadata=spec.user_metadata or {},
                owner_id=owner_id,
                mime_type=spec.mime_type,
                preview_id=spec.preview_id,
            )
            if result is None:
                delete_temp_file_if_exists(parsed.tmp_path)
                return _build_error_response(
                    404, "ASSET_NOT_FOUND", f"Asset content {spec.hash} does not exist"
                )
            delete_temp_file_if_exists(parsed.tmp_path)
        else:
            # Otherwise, we must have a temp file path to ingest
            if not parsed.tmp_path or not os.path.exists(parsed.tmp_path):
                return _build_error_response(
                    400,
                    "MISSING_INPUT",
                    "Provided hash not found and no file uploaded.",
                )

            result = upload_from_temp_path(
                temp_path=parsed.tmp_path,
                name=spec.name,
                tags=spec.tags,
                user_metadata=spec.user_metadata or {},
                client_filename=parsed.file_client_name,
                owner_id=owner_id,
                expected_hash=spec.hash,
                mime_type=spec.mime_type,
                preview_id=spec.preview_id,
            )
    except AssetValidationError as e:
        delete_temp_file_if_exists(parsed.tmp_path)
        return _build_error_response(400, e.code, str(e))
    except ValueError as e:
        delete_temp_file_if_exists(parsed.tmp_path)
        return _build_error_response(400, "BAD_REQUEST", str(e))
    except HashMismatchError as e:
        delete_temp_file_if_exists(parsed.tmp_path)
        return _build_error_response(400, "HASH_MISMATCH", str(e))
    except DependencyMissingError as e:
        delete_temp_file_if_exists(parsed.tmp_path)
        return _build_error_response(503, "DEPENDENCY_MISSING", e.message)
    except Exception:
        delete_temp_file_if_exists(parsed.tmp_path)
        logging.exception("upload_asset failed for owner_id=%s", owner_id)
        return _build_error_response(500, "INTERNAL", "Unexpected server error.")

    asset = _build_asset_response(result)
    payload_out = schemas_out.AssetCreated(
        **asset.model_dump(),
        created_new=result.created_new,
    )
    status = 201 if result.created_new else 200
    return web.json_response(payload_out.model_dump(mode="json", exclude_none=True), status=status)


async def _handle_chunked_upload(request, parsed, spec, owner_id):
    """Handle chunked file upload with progress tracking."""
    upload_id = spec.upload_id
    
    # Initialize upload session if first chunk
    if upload_id not in _UPLOAD_SESSIONS:
        _UPLOAD_SESSIONS[upload_id] = {
            "chunks": {},
            "total_chunks": spec.total_chunks,
            "metadata": {
                "tags": spec.tags,
                "name": spec.name,
                "user_metadata": spec.user_metadata,
                "mime_type": spec.mime_type,
                "preview_id": spec.preview_id,
                "hash": spec.hash,
            },
            "created_at": datetime.now(),
            "owner_id": owner_id,
        }
    
    session = _UPLOAD_SESSIONS[upload_id]
    
    # Validate chunk number
    if spec.chunk_number > spec.total_chunks:
        delete_temp_file_if_exists(parsed.tmp_path)
        return _build_error_response(400, "INVALID_BODY", "Chunk number exceeds total chunks")
    
    # Store chunk
    if parsed.tmp_path and os.path.exists(parsed.tmp_path):
        session["chunks"][spec.chunk_number] = parsed.tmp_path
    else:
        return _build_error_response(400, "MISSING_INPUT", "No file data in chunk")
    
    # Calculate progress
    progress = len(session["chunks"]) / session["total_chunks"] * 100
    
    # Check if all chunks received
    if len(session["chunks"]) == session["total_chunks"]:
        # Combine chunks and complete upload
        return await _complete_chunked_upload(upload_id, session)
    
    # Return progress
    return web.json_response({
        "status": "uploading",
        "upload_id": upload_id,
        "chunk_number": spec.chunk_number,
        "total_chunks": spec.total_chunks,
        "progress": round(progress, 2),
    }, status=200)


async def _complete_chunked_upload(upload_id, session):
    """Combine all chunks and complete the upload."""
    import tempfile
    import shutil
    
    try:
        # Create combined file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".combined") as combined_file:
            combined_path = combined_file.name
            
            # Combine chunks in order
            for chunk_num in sorted(session["chunks"].keys()):
                chunk_path = session["chunks"][chunk_num]
                with open(chunk_path, "rb") as chunk_file:
                    shutil.copyfileobj(chunk_file, combined_file)
                os.unlink(chunk_path)  # Clean up chunk
        
        # Upload combined file
        metadata = session["metadata"]
        result = upload_from_temp_path(
            temp_path=combined_path,
            name=metadata["name"],
            tags=metadata["tags"],
            user_metadata=metadata["user_metadata"],
            client_filename=f"chunked_upload_{upload_id}",
            owner_id=session["owner_id"],
            expected_hash=metadata["hash"],
            mime_type=metadata["mime_type"],
            preview_id=metadata["preview_id"],
        )
        
        # Clean up
        os.unlink(combined_path)
        del _UPLOAD_SESSIONS[upload_id]
        
        asset = _build_asset_response(result)
        payload_out = schemas_out.AssetCreated(
            **asset.model_dump(),
            created_new=result.created_new,
        )
        status = 201 if result.created_new else 200
        return web.json_response(payload_out.model_dump(mode="json", exclude_none=True), status=status)
        
    except Exception as e:
        logging.exception("Failed to complete chunked upload %s", upload_id)
        # Clean up session
        if upload_id in _UPLOAD_SESSIONS:
            for chunk_path in _UPLOAD_SESSIONS[upload_id]["chunks"].values():
                if os.path.exists(chunk_path):
                    os.unlink(chunk_path)
            del _UPLOAD_SESSIONS[upload_id]
        return _build_error_response(500, "INTERNAL", f"Failed to complete chunked upload: {str(e)}")


@ROUTES.put(f"/api/assets/{{id:{UUID_RE}}}")
@_require_assets_feature_enabled
async def update_asset_route(request: web.Request) -> web.Response:
    reference_id = str(uuid.UUID(request.match_info["id"]))
    owner_id = USER_MANAGER.get_request_user_id(request)
    
    try:
        body = schemas_in.UpdateAssetBody.model_validate(await request.json())
    except ValidationError as ve:
        return _build_validation_error_response("INVALID_BODY", ve)
    except Exception:
        return _build_error_response(
            400, "INVALID_JSON", "Request body must be valid JSON."
        )

    try:
        # Get current state for undo
        current_result = get_asset_detail(reference_id=reference_id, owner_id=owner_id)
        previous_state = _build_asset_response(current_result).model_dump(mode="json") if current_result else None
        
        result = update_asset_metadata(
            reference_id=reference_id,
            name=body.name,
            user_metadata=body.user_metadata,
            owner_id=owner_id,
            preview_id=body.preview_id,
        )
        payload = _build_asset_response(result)
        
        # Record operation for undo
        if previous_state:
            _record_operation(owner_id, "update", reference_id, previous_state)
        
        # Invalidate cache for this asset
        _invalidate_cache(reference_id)
        
    except PermissionError as pe:
        return _build_error_response(403, "FORBIDDEN", str(pe), {"id": reference_id})
    except ValueError as ve:
        return _build_error_response(
            404, "ASSET_NOT_FOUND", str(ve), {"id": reference_id}
        )
    except Exception:
        logging.exception(
            "update_asset failed for reference_id=%s, owner_id=%s",
            reference_id,
            owner_id,
        )
        return _build_error_response(500, "INTERNAL", "Unexpected server error.")
    return web.json_response(payload.model_dump(mode="json", exclude_none=True), status=200)


@ROUTES.delete(f"/api/assets/{{id:{UUID_RE}}}")
@_require_assets_feature_enabled
async def delete_asset_route(request: web.Request) -> web.Response:
    reference_id = str(uuid.UUID(request.match_info["id"]))
    delete_content_param = request.query.get("delete_content")
    delete_content = (
        False
        if delete_content_param is None
        else delete_content_param.lower() not in {"0", "false", "no"}
    )

    try:
        deleted = delete_asset_reference(
            reference_id=reference_id,
            owner_id=USER_MANAGER.get_request_user_id(request),
            delete_content_if_orphan=delete_content,
        )
    except Exception:
        logging.exception(
            "delete_asset_reference failed for reference_id=%s, owner_id=%s",
            reference_id,
            USER_MANAGER.get_request_user_id(request),
        )
        return _build_error_response(500, "INTERNAL", "Unexpected server error.")

    if not deleted:
        return _build_error_response(
            404, "ASSET_NOT_FOUND", f"AssetReference {reference_id} not found."
        )
    return web.Response(status=204)


@ROUTES.get("/api/tags")
@_require_assets_feature_enabled
async def get_tags(request: web.Request) -> web.Response:
    """
    GET request to list all tags based on query parameters.
    """
    query_map = dict(request.rel_url.query)

    try:
        query = schemas_in.TagsListQuery.model_validate(query_map)
    except ValidationError as e:
        return _build_error_response(
            400,
            "INVALID_QUERY",
            "Invalid query parameters",
            {"errors": json.loads(e.json())},
        )

    rows, total = list_tags(
        prefix=query.prefix,
        limit=query.limit,
        offset=query.offset,
        order=query.order,
        include_zero=query.include_zero,
        owner_id=USER_MANAGER.get_request_user_id(request),
    )

    tags = [
        schemas_out.TagUsage(name=name, count=count, type=tag_type)
        for (name, tag_type, count) in rows
    ]
    payload = schemas_out.TagsList(
        tags=tags, total=total, has_more=(query.offset + len(tags)) < total
    )
    return web.json_response(payload.model_dump(mode="json", exclude_none=True))


@ROUTES.post(f"/api/assets/{{id:{UUID_RE}}}/tags")
@_require_assets_feature_enabled
async def add_asset_tags(request: web.Request) -> web.Response:
    reference_id = str(uuid.UUID(request.match_info["id"]))
    try:
        json_payload = await request.json()
        data = schemas_in.TagsAdd.model_validate(json_payload)
    except ValidationError as ve:
        return _build_error_response(
            400,
            "INVALID_BODY",
            "Invalid JSON body for tags add.",
            {"errors": ve.errors()},
        )
    except Exception:
        return _build_error_response(
            400, "INVALID_JSON", "Request body must be valid JSON."
        )

    try:
        result = apply_tags(
            reference_id=reference_id,
            tags=data.tags,
            origin="manual",
            owner_id=USER_MANAGER.get_request_user_id(request),
        )
        payload = schemas_out.TagsAdd(
            added=result.added,
            already_present=result.already_present,
            total_tags=result.total_tags,
        )
    except PermissionError as pe:
        return _build_error_response(403, "FORBIDDEN", str(pe), {"id": reference_id})
    except ValueError as ve:
        return _build_error_response(
            404, "ASSET_NOT_FOUND", str(ve), {"id": reference_id}
        )
    except Exception:
        logging.exception(
            "add_tags_to_asset failed for reference_id=%s, owner_id=%s",
            reference_id,
            USER_MANAGER.get_request_user_id(request),
        )
        return _build_error_response(500, "INTERNAL", "Unexpected server error.")

    return web.json_response(payload.model_dump(mode="json", exclude_none=True), status=200)


@ROUTES.delete(f"/api/assets/{{id:{UUID_RE}}}/tags")
@_require_assets_feature_enabled
async def delete_asset_tags(request: web.Request) -> web.Response:
    reference_id = str(uuid.UUID(request.match_info["id"]))
    try:
        json_payload = await request.json()
        data = schemas_in.TagsRemove.model_validate(json_payload)
    except ValidationError as ve:
        return _build_error_response(
            400,
            "INVALID_BODY",
            "Invalid JSON body for tags remove.",
            {"errors": ve.errors()},
        )
    except Exception:
        return _build_error_response(
            400, "INVALID_JSON", "Request body must be valid JSON."
        )

    try:
        result = remove_tags(
            reference_id=reference_id,
            tags=data.tags,
            owner_id=USER_MANAGER.get_request_user_id(request),
        )
        payload = schemas_out.TagsRemove(
            removed=result.removed,
            not_present=result.not_present,
            total_tags=result.total_tags,
        )
    except PermissionError as pe:
        return _build_error_response(403, "FORBIDDEN", str(pe), {"id": reference_id})
    except ValueError as ve:
        return _build_error_response(
            404, "ASSET_NOT_FOUND", str(ve), {"id": reference_id}
        )
    except Exception:
        logging.exception(
            "remove_tags_from_asset failed for reference_id=%s, owner_id=%s",
            reference_id,
            USER_MANAGER.get_request_user_id(request),
        )
        return _build_error_response(500, "INTERNAL", "Unexpected server error.")

    return web.json_response(payload.model_dump(mode="json", exclude_none=True), status=200)


@ROUTES.get("/api/assets/tags/refine")
@_require_assets_feature_enabled
async def get_tags_refine(request: web.Request) -> web.Response:
    """GET request to get tag histogram for filtered assets."""
    query_dict = get_query_dict(request)
    try:
        q = schemas_in.TagsRefineQuery.model_validate(query_dict)
    except ValidationError as ve:
        return _build_validation_error_response("INVALID_QUERY", ve)

    tag_counts = list_tag_histogram(
        owner_id=USER_MANAGER.get_request_user_id(request),
        include_tags=q.include_tags,
        exclude_tags=q.exclude_tags,
        name_contains=q.name_contains,
        metadata_filter=q.metadata_filter,
        limit=q.limit,
    )
    payload = schemas_out.TagHistogram(tag_counts=tag_counts)
    return web.json_response(payload.model_dump(mode="json", exclude_none=True), status=200)


@ROUTES.post("/api/assets/seed")
@_require_assets_feature_enabled
async def seed_assets(request: web.Request) -> web.Response:
    """Trigger asset seeding for specified roots (models, input, output).

    Query params:
        wait: If "true", block until scan completes (synchronous behavior for tests)

    Returns:
        202 Accepted if scan started
        409 Conflict if scan already running
        200 OK with final stats if wait=true
    """
    try:
        payload = await request.json()
        roots = payload.get("roots", ["models", "input", "output"])
    except Exception:
        roots = ["models", "input", "output"]

    valid_roots = tuple(r for r in roots if r in ("models", "input", "output"))
    if not valid_roots:
        return _build_error_response(400, "INVALID_BODY", "No valid roots specified")

    wait_param = request.query.get("wait", "").lower()
    should_wait = wait_param in ("true", "1", "yes")

    started = asset_seeder.start(roots=valid_roots)
    if not started:
        return web.json_response({"status": "already_running"}, status=409)

    if should_wait:
        await asyncio.to_thread(asset_seeder.wait)
        status = asset_seeder.get_status()
        return web.json_response(
            {
                "status": "completed",
                "progress": {
                    "scanned": status.progress.scanned if status.progress else 0,
                    "total": status.progress.total if status.progress else 0,
                    "created": status.progress.created if status.progress else 0,
                    "skipped": status.progress.skipped if status.progress else 0,
                },
                "errors": status.errors,
            },
            status=200,
        )

    return web.json_response({"status": "started"}, status=202)


@ROUTES.get("/api/assets/seed/status")
@_require_assets_feature_enabled
async def get_seed_status(request: web.Request) -> web.Response:
    """Get current scan status and progress."""
    status = asset_seeder.get_status()
    return web.json_response(
        {
            "state": status.state.value,
            "progress": {
                "scanned": status.progress.scanned,
                "total": status.progress.total,
                "created": status.progress.created,
                "skipped": status.progress.skipped,
            }
            if status.progress
            else None,
            "errors": status.errors,
        },
        status=200,
    )


@ROUTES.post("/api/assets/seed/cancel")
@_require_assets_feature_enabled
async def cancel_seed(request: web.Request) -> web.Response:
    """Request cancellation of in-progress scan."""
    cancelled = asset_seeder.cancel()
    if cancelled:
        return web.json_response({"status": "cancelling"}, status=200)
    return web.json_response({"status": "idle"}, status=200)


@ROUTES.post("/api/assets/undo")
@_require_assets_feature_enabled
async def undo_last_operation(request: web.Request) -> web.Response:
    """Undo the last asset operation for the current user."""
    owner_id = USER_MANAGER.get_request_user_id(request)
    
    last_op = _undo_operation(owner_id)
    if not last_op:
        return _build_error_response(404, "NO_OPERATION", "No operation to undo")
    
    try:
        if last_op["operation"] == "update":
            # Restore previous state
            result = update_asset_metadata(
                reference_id=last_op["asset_id"],
                name=last_op["previous_state"].get("name"),
                user_metadata=last_op["previous_state"].get("user_metadata"),
                owner_id=owner_id,
                preview_id=last_op["previous_state"].get("preview_id"),
            )
            _invalidate_cache(last_op["asset_id"])
            payload = _build_asset_response(result)
            return web.json_response({
                "message": "Operation undone successfully",
                "asset": payload.model_dump(mode="json", exclude_none=True)
            }, status=200)
        else:
            return _build_error_response(400, "UNSUPPORTED_OPERATION", f"Cannot undo operation: {last_op['operation']}")
    except Exception as e:
        logging.exception("Failed to undo operation")
        return _build_error_response(500, "INTERNAL", f"Failed to undo operation: {str(e)}")


@ROUTES.get("/api/assets/operations/history")
@_require_assets_feature_enabled
async def get_operation_history(request: web.Request) -> web.Response:
    """Get operation history for the current user."""
    owner_id = USER_MANAGER.get_request_user_id(request)
    history = _OPERATION_HISTORY.get(owner_id, [])
    
    return web.json_response({
        "history": [
            {
                "operation": op["operation"],
                "asset_id": op["asset_id"],
                "timestamp": op["timestamp"].isoformat()
            }
            for op in history
        ],
        "count": len(history)
    }, status=200)


@ROUTES.post("/api/assets/prune")
@_require_assets_feature_enabled
async def mark_missing_assets(request: web.Request) -> web.Response:
    """Mark assets as missing when outside all known root prefixes.

    This is a non-destructive soft-delete operation. Assets and metadata
    are preserved, but references are flagged as missing. They can be
    restored if the file reappears in a future scan.

    Returns:
        200 OK with count of marked assets
        409 Conflict if a scan is currently running
    """
    try:
        marked = asset_seeder.mark_missing_outside_prefixes()
    except ScanInProgressError:
        return web.json_response(
            {"status": "scan_running", "marked": 0},
            status=409,
        )
    return web.json_response({"status": "completed", "marked": marked}, status=200)


@ROUTES.get("/api/assets/seed/progress")
@_require_assets_feature_enabled
async def seed_progress_stream(request: web.Request) -> web.Response:
    """Server-Sent Events endpoint for real-time asset seeding progress.

    Returns:
        200 OK with SSE stream
        409 Conflict if no scan is running
    """
    response = web.StreamResponse()
    response.content_type = "text/event-stream"
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Connection"] = "keep-alive"
    response.headers["X-Accel-Buffering"] = "no"
    
    await response.prepare(request)
    
    try:
        last_status = None
        while True:
            status = asset_seeder.get_status()
            
            # Only send if status changed
            if status.state.value != "idle" and status != last_status:
                progress_data = {
                    "state": status.state.value,
                    "progress": {
                        "scanned": status.progress.scanned if status.progress else 0,
                        "total": status.progress.total if status.progress else 0,
                        "created": status.progress.created if status.progress else 0,
                        "skipped": status.progress.skipped if status.progress else 0,
                        "percentage": round((status.progress.scanned / status.progress.total * 100) if status.progress and status.progress.total > 0 else 0, 2),
                    } if status.progress else None,
                    "errors": status.errors,
                }
                
                await response.write(f"data: {json.dumps(progress_data)}\n\n".encode())
                last_status = status
            
            # Exit if scan is complete
            if status.state.value == "idle":
                await response.write(f"data: {json.dumps({'state': 'completed'})}\n\n".encode())
                break
            
            await asyncio.sleep(0.5)
            
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logging.exception("Error in progress stream")
        await response.write(f"data: {json.dumps({'error': str(e)})}\n\n".encode())
    
    return response
