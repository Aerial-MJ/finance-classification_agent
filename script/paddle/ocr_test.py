# Initialize PaddleOCR instance
# from paddleocr import PaddleOCR
# ocr = PaddleOCR(
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=False,
#     use_textline_orientation=False)

# # Run OCR inference on a sample image 
# result = ocr.predict(
#     input="./image.png")

# # Visualize the results and save the JSON results
# for res in result:
#     res.print()
#     res.save_to_img("output1")
#     res.save_to_json("output1")

from paddleocr import DocPreprocessor,PaddleOCR
from PIL import Image

# 初始化预处理产线
def rotate_Image(image_path):

    ocr = PaddleOCR(
        use_doc_orientation_classify=True,
        use_doc_unwarping=False,
        use_textline_orientation=True
    )
    result = ocr.predict(input=image_path, return_word_box=True)
    for res in result:
        angle=res.json["res"]["angle"]

    img = Image.open(image_path)
    rotated = img.rotate(angle)

