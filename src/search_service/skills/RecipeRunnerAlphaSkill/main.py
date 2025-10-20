import azure.functions as func
import json
import logging
import traceback
from .models.request import SkillRequest
from .recipe_runner import RecipeRunner
from .registry import RECIPE_REGISTRY


async def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        # Retrieve the recipe ID from the HTTP headers (ensure it is provided)
        recipe_id = req.headers.get("Recipe-Id")
        print(f"Received request for Recipe ID: {recipe_id}")
        if not recipe_id:
            raise ValueError("Missing 'Recipe-Id' header")

        # Parse the incoming JSON body
        try:
            data = req.get_json()
        except Exception as json_err:
            raise ValueError("Invalid JSON body") from json_err

        if "values" not in data:
            raise ValueError("Missing 'values' key in request body")

        # Convert each dictionary in 'values' into a SkillRequest instance
        try:
            skill_requests = [SkillRequest(**item) for item in data["values"]]
        except Exception as conversion_err:
            raise ValueError(
                "Error converting input data to SkillRequest objects"
            ) from conversion_err
        
        if recipe_id not in RECIPE_REGISTRY:
            raise ValueError(f"Recipe ID '{recipe_id}' not found in registry")

        # Create a RecipeRunner instance with the provided recipe ID and requests
        recipe_runner = RecipeRunner(recipe_id=recipe_id, skill_requests=skill_requests)

        # Process all files concurrently and obtain the results
        results = await recipe_runner.process_all()

        # Return the processing results as a JSON response
        return func.HttpResponse(
            body=json.dumps(results), mimetype="application/json", status_code=200
        )
    except Exception as e:
        # Log the complete stack trace for diagnostic purposes
        logging.exception("An error occurred while processing the request.")

        # Optionally, include stack trace in the HTTP response if not in production.
        # For production, it is recommended to only return a generic error message.
        error_response = {
            "error": str(e),
            "trace": traceback.format_exc(),  # Remove or conditionally include in production
        }

        return func.HttpResponse(
            body=json.dumps(error_response),
            mimetype="application/json",
            status_code=500,
        )
