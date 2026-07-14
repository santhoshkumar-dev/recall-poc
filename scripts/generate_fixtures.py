from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "sample-data"
OUT.mkdir(parents=True, exist_ok=True)


def font(size: int):
    candidates = [
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def make_ocr_image(path: Path, heading: str, lines: list[str], fmt: str):
    image = Image.new("RGB", (1400, 900), "#f8f6ee")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((60, 60, 1340, 840), radius=36, fill="#ffffff", outline="#d7d2c5", width=4)
    draw.rectangle((60, 60, 1340, 190), fill="#171714")
    draw.text((110, 100), heading, font=font(50), fill="#c7ff4a")
    y = 250
    for line in lines:
        draw.text((110, y), line, font=font(38), fill="#171714")
        y += 82
    image.save(path, format=fmt, quality=94)


def make_pdf(path: Path):
    pdf = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4

    def header(label: str, page: int):
        pdf.setFillColor(HexColor("#171714"))
        pdf.rect(0, height - 105, width, 105, stroke=0, fill=1)
        pdf.setFillColor(HexColor("#c7ff4a"))
        pdf.setFont("Helvetica-Bold", 25)
        pdf.drawString(48, height - 68, label)
        pdf.setFillColor(HexColor("#77776f"))
        pdf.setFont("Helvetica", 9)
        pdf.drawRightString(width - 48, 32, f"Recall synthetic fixture - page {page}")

    header("NIGHT TRAIN TICKET", 1)
    pdf.setFillColor(HexColor("#171714"))
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(48, height - 165, "Chennai Central to Bengaluru")
    fields = [
        ("Departure", "Friday, 22:15"),
        ("Arrival", "Saturday, 05:50"),
        ("Booking reference", "RCL-4821"),
        ("Coach and seat", "A2 / 31"),
    ]
    y = height - 220
    for label, value in fields:
        pdf.setFillColor(HexColor("#77776f")); pdf.setFont("Helvetica", 10); pdf.drawString(48, y, label.upper())
        pdf.setFillColor(HexColor("#171714")); pdf.setFont("Helvetica-Bold", 14); pdf.drawString(190, y - 2, value)
        y -= 55
    pdf.setFillColor(HexColor("#315a3a")); pdf.roundRect(48, y - 70, width - 96, 70, 12, stroke=0, fill=1)
    pdf.setFillColor(HexColor("#ffffff")); pdf.setFont("Helvetica", 12); pdf.drawString(70, y - 42, "Carry a printed identity document and arrive 30 minutes early.")
    pdf.showPage()

    header("PURCHASE RECEIPT", 2)
    pdf.setFillColor(HexColor("#171714")); pdf.setFont("Helvetica-Bold", 18); pdf.drawString(48, height - 165, "Northstar Electronics")
    pdf.setFont("Helvetica", 12)
    receipt = [
        "Portable SSD - 2 TB", "Purchase date: 12 July 2026", "Total paid: INR 12,499",
        "Warranty expires: 12 July 2029", "Receipt number: NS-88420",
    ]
    y = height - 220
    for line in receipt:
        pdf.drawString(48, y, line); y -= 38
    pdf.setFillColor(HexColor("#f1f0e9")); pdf.roundRect(48, y - 90, width - 96, 90, 12, stroke=0, fill=1)
    pdf.setFillColor(HexColor("#171714")); pdf.setFont("Helvetica-Bold", 12); pdf.drawString(70, y - 38, "Reminder")
    pdf.setFont("Helvetica", 11); pdf.drawString(70, y - 62, "Keep this receipt for the three-year warranty claim.")
    pdf.save()


make_pdf(OUT / "ticket-and-receipt.pdf")
make_ocr_image(OUT / "event-poster.png", "BENGALURU DESIGN NIGHT", ["Saturday 18 July", "6:30 PM at Indiranagar", "Talks, demos, and local food", "Entry code: RECALL"], "PNG")
make_ocr_image(OUT / "restaurant-card.jpg", "ANIKA RECOMMENDS", ["Try Green Pepper Kitchen", "12 Church Street, Bengaluru", "Order the mushroom pepper fry", "Open until 10:30 PM"], "JPEG")
make_ocr_image(OUT / "warranty-card.webp", "CAMERA WARRANTY", ["Model: PocketCam X2", "Serial: PCX2-7719", "Coverage ends 03 March 2028", "Keep the purchase invoice"], "WEBP")
print(f"Generated fixtures in {OUT}")
