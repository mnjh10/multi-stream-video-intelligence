from fastapi import APIRouter

from backend.schemas.query import QueryRequest, QueryResponse, QueryResult
from backend.services.retrieval import MockRetrievalProvider


router = APIRouter()

retrieval_provider = MockRetrievalProvider()


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    results = retrieval_provider.search(request.query)

    return QueryResponse(
        query=request.query,
        status="ok",
        results=[
            QueryResult(**result)
            for result in results
        ],
    )