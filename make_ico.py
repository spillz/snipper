from PIL import Image
img = Image.open(r"images\snipper-icon.png")
img.save(r"images\snipper-icon.ico", sizes=[(16,16),(32,32),(48,48),(64,64),(128,128),(256,256)])
