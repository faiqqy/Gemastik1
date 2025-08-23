import qrcode
from PIL import Image,ImageDraw,ImageFont


def bikinQr(result):
    qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=8, border=4) 

    img = qr.add_data(result)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    img = qrcode.make(result)
    img = img.resize((128, 128)).convert('RGB')

    img.save("qr.png")
    
bikinQr("2YtylQxSpZM2AFTt7tKSE8DNc3+3Pzuxq1g5yKyJINXM7IVrW0T/bf4zyt1Klca1XXzko08kY/nRneq9u7UXQ+8j+mCpyObhbpxRTN0=:hGZUhDSiI2Lv75TPoN4kvA==")