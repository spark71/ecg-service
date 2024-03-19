from fastapi import APIRouter, Request, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from app.pages.form_schema import DataForm


router = APIRouter(
    prefix='/pages',
    tags=['Фронтенд']
)

templates = Jinja2Templates(directory='app/templates')


@router.get('/add_sig', response_class=HTMLResponse)
async def get_add_sig(request: Request):
    return templates.TemplateResponse(name="add_form.html", context={'request': request})



@router.post('/add_sig', response_class=HTMLResponse)
async def post_add_sig(request: Request, form_data: DataForm = Depends(DataForm.as_form)):
    # print(dict(form_data))
    # print(form_data.id)
    # print(form_data.leads_values)
    data = await form_data.leads_values.read()
    await form_data.leads_values.close()
    print(data)
    return templates.TemplateResponse(name="add_form.html", context={'request': request})
