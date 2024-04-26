# server.py
from fastapi import FastAPI
from mangum import Mangum
from fastapi.responses import RedirectResponse
from langserve import add_routes

from anthropic_iterative_search import chain as anthropic_iterative_search_chain

app = FastAPI()

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

add_routes(app, anthropic_iterative_search_chain, path="/anthropic-iterative-search")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

handler = Mangum(app)



# from fastapi import FastAPI
# from fastapi.responses import RedirectResponse
# from langserve import add_routes
# from anthropic_iterative_search import chain as anthropic_iterative_search_chain

# app = FastAPI()

# @app.get("/")
# async def redirect_root_to_docs():
#     return RedirectResponse("/docs")


# # Edit this to add the chain you want to add
# add_routes(app, anthropic_iterative_search_chain, path="/packages/anthropic-iterative-search")

# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="0.0.0.0", port=8000)
