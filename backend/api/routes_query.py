from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.db.models import Query, SemanticMemory
from backend.db.session import get_db
from backend.schemas.query import QueryRequest, QueryResponse, QueryResult
from backend.services.retrieval import RetrievalProvider, get_retrieval_provider
from backend.services.temporal import get_temporal_followup_engine
from retrieval.temporal_followup import TemporalFollowupEngine

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    db: Session = Depends(get_db),
    provider: RetrievalProvider = Depends(get_retrieval_provider),
    temporal_engine: TemporalFollowupEngine = Depends(get_temporal_followup_engine),
):
    # Check for empty or whitespace query
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    ctx_dict = request.context.model_dump() if request.context else None

    # Check for temporal follow-up query
    if temporal_engine.is_followup_query(request.query, context=ctx_dict):
        followup_res = temporal_engine.resolve_followup(
            query=request.query,
            context=ctx_dict,
            top_k=request.top_k,
        )

        try:
            db_query = Query(query_text=request.query, status="completed")
            db.add(db_query)
            db.commit()
        except Exception:
            db.rollback()

        return QueryResponse(
            query=request.query,
            status=followup_res.get("status", "ok"),
            message=followup_res.get("message"),
            results=[
                QueryResult(**r)
                for r in followup_res.get("results", [])
            ],
        )

    # 1. Resolve semantic memory aliases if present
    filters = dict(request.filters or {})
    if "camera_id" not in filters:
        try:
            memories = db.scalars(select(SemanticMemory)).all()
            q_lower = request.query.lower()
            for mem in memories:
                if mem.key and mem.key.lower() in q_lower and mem.camera_id:
                    filters["camera_id"] = mem.camera_id
                    break
        except Exception:
            pass

    # 2. Execute retrieval
    try:
        results = provider.search(
            query=request.query,
            filters=filters if filters else None,
            top_k=request.top_k,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Enrich results with observation metadata if not already attached by provider
    for res in results:
        cam_id = res.get("camera_id")
        obj_id = res.get("object_id")
        ts = res.get("best_timestamp") if res.get("best_timestamp") is not None else res.get("timestamp")
        if cam_id and ts is not None and not res.get("bbox"):
            try:
                obs = temporal_engine.find_observation(cam_id, obj_id, float(ts))
                if obs:
                    res["bbox"] = obs.get("bbox")
                    res["frame_index"] = obs.get("frame_index")
                    res["crop_path"] = obs.get("crop_path")
                    if not res.get("observation_id"):
                        res["observation_id"] = obs.get("observation_id")
            except Exception:
                pass

    # Update last context to top result
    if results:
        top = results[0]
        temporal_engine.set_last_context({
            "camera_id": top.get("camera_id"),
            "object_id": top.get("object_id"),
            "timestamp": top.get("best_timestamp") or top.get("timestamp"),
            "best_timestamp": top.get("best_timestamp") or top.get("timestamp"),
            "event_id": top.get("event_id"),
            "bbox": top.get("bbox"),
            "frame_index": top.get("frame_index"),
        })

    # 3. Record query in DB
    try:
        db_query = Query(query_text=request.query, status="completed")
        db.add(db_query)
        db.commit()
    except Exception:
        db.rollback()

    # Determine status and feedback message
    response_status = "ok"
    response_message = None

    if not results:
        from retrieval.query_parser import QueryParser
        parsed = QueryParser().parse(request.query)
        attrs = parsed.get("attributes", [])
        cam = parsed.get("camera_id")
        obj_type = parsed.get("object_type")
        if attrs:
            attr_str = " ".join(attrs)
            target_str = f"{attr_str} {obj_type or 'object'}"
            cam_str = f"in camera {cam}" if cam else "across all cameras"
            response_status = "no_match"
            response_message = f"No events matching '{target_str}' verified {cam_str}."
        elif cam and obj_type:
            response_status = "no_match"
            response_message = f"No events for '{obj_type}' found in camera {cam}."

    # 4. Return formatted response
    return QueryResponse(
        query=request.query,
        status=response_status,
        message=response_message,
        results=[
            QueryResult(**result)
            for result in results
        ],
    )