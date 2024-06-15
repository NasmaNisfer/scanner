import cv2
import imutils
from skimage.filters import threshold_local
from transform import perspective_transform
from fastapi import FastAPI, File, UploadFile
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


@app.get("/static", response_class=HTMLResponse)
def list_files(request: Request):

    files = os.listdir("./static")
    files_paths = sorted([f"{request.url._url}/{f}" for f in files])
    print(files_paths)
    return templates.TemplateResponse(
        "list_files.html", {"request": request, "files": files_paths}
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/ssebowaAI-scanner")
def mainAPI(img: UploadFile = File(None)):
    name = img.filename
    contents = img.file.read()
    with open(name,'wb') as data:
        data.write(contents)
        data.close()

    # Passing the image path
    original_img = cv2.imread(name)
    copy = original_img.copy()

    # The resized height in hundreds
    ratio = original_img.shape[0] / 500.0
    img_resize = imutils.resize(original_img, height=500)

    gray_image = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)

    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edged_img = cv2.Canny(blurred_image, 75, 200)

    cnts, _ = cv2.findContours(edged_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            doc = approx
            break

    p = []

    for d in doc:
        tuple_point = tuple(d[0])
        cv2.circle(img_resize, tuple_point, 3, (0, 0, 255), 4)
        p.append(tuple_point)

    warped_image = perspective_transform(copy, doc.reshape(4, 2) * ratio)
    warped_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)

    T = threshold_local(warped_image,21, offset=10, method="gaussian")
    warped = (warped_image > T).astype("uint8") * 255
    cv2.imwrite('scan '+name,warped)

#uvicorn scan:app
