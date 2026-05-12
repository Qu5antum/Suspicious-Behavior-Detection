from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.api.endpoint import event_router


app = FastAPI(
    title="Event Manager",
    debug=True,
    docs_url="/docs",
)

app.mount("/events_file", StaticFiles(directory="events_file"), name="events_file")


app.add_middleware(
    CORSMiddleware,
    allow_origins = ['*'],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)

app.include_router(event_router)


if __name__ == "__main__":
    uvicorn.run(
        "src.run_api:app", host="127.0.0.1", port=8000, reload=True 
)