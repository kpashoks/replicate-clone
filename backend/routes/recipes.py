"""Named recipes: saved (model + params + input references) configs that
let the user re-run a generation later with one click.

A recipe is a JSON file under data/recipes/<recipe_id>.json holding
everything needed to repopulate the task form:
  - slug + task (which model / task page)
  - params (prompt, seed, dynamic-form values, lora, etc.)
  - input_ids (references to uploaded files in data/inputs/; persistent
    content-addressed, so they survive as long as the files aren't
    cleared)

Recipes are independent of jobs. Saving a recipe doesn't run anything;
loading one just repopulates the form. The user then clicks Run.
"""

from __future__ import annotations

import json
import time
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import storage


router = APIRouter(prefix="/api/recipes", tags=["recipes"])


class RecipeCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    slug: str = Field(..., min_length=1)
    task: str = Field(..., min_length=1)
    params: dict = Field(default_factory=dict)
    input_ids: list[str] = Field(default_factory=list)
    # Optional preview: the output-file path of the job that was on screen
    # when the recipe was saved (e.g. "outputs/<job_id>/0.png"). Rendered
    # as a thumbnail in the recipe list. None if saved before any run.
    thumbnail: str | None = None


class Recipe(RecipeCreate):
    id: str
    created_at: float


def _recipe_path(recipe_id: str):
    # Guard against path traversal: recipe_id must be a bare hex token.
    if not recipe_id or "/" in recipe_id or "\\" in recipe_id or ".." in recipe_id:
        raise HTTPException(status_code=400, detail="Invalid recipe id")
    return storage.recipes_dir() / f"{recipe_id}.json"


def _load(recipe_id: str) -> Recipe:
    p = _recipe_path(recipe_id)
    if not p.is_file():
        raise HTTPException(status_code=404, detail="Recipe not found")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise HTTPException(status_code=500, detail=f"Failed to read recipe: {e}")
    return Recipe(**data)


@router.post("", response_model=Recipe)
async def create_recipe(body: RecipeCreate) -> Recipe:
    recipe = Recipe(
        id=uuid.uuid4().hex[:12],
        created_at=time.time(),
        **body.model_dump(),
    )
    p = _recipe_path(recipe.id)
    try:
        p.write_text(
            json.dumps(recipe.model_dump(), indent=2), encoding="utf-8",
        )
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Failed to save recipe: {e}")
    return recipe


@router.get("", response_model=list[Recipe])
async def list_recipes(task: str | None = None) -> list[Recipe]:
    """List all recipes, newest first. Optional ?task= filter so the task
    page only shows recipes relevant to it."""
    out: list[Recipe] = []
    for p in storage.recipes_dir().glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            r = Recipe(**data)
        except (OSError, ValueError, TypeError):
            continue  # skip corrupt/old-format files rather than 500
        if task is not None and r.task != task:
            continue
        out.append(r)
    out.sort(key=lambda r: r.created_at, reverse=True)
    return out


@router.get("/{recipe_id}", response_model=Recipe)
async def get_recipe(recipe_id: str) -> Recipe:
    return _load(recipe_id)


@router.delete("/{recipe_id}")
async def delete_recipe(recipe_id: str) -> dict:
    p = _recipe_path(recipe_id)
    if not p.is_file():
        raise HTTPException(status_code=404, detail="Recipe not found")
    try:
        p.unlink()
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete recipe: {e}")
    return {"deleted": recipe_id}
