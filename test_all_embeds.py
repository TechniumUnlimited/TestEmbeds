from PIL import Image, PngImagePlugin
import requests, os, base64, io

def get_embeds(embed_folder_Path):
    # Get every filename without the extension in the folder
    embed_names = []
    for f in os.listdir(embed_folder_Path):
        if f[:-4] not in embed_names:
            embed_names.append(f[:-4])

# Send a prompt to stable diffusion api to get the images
def prompt_stablediffusion(prompt, steps=50, iter=1):
    url = "http://127.0.0.1:7860"

    payload = {
        "prompt": prompt,
        "steps": steps,
        "enable_hr": False,
        "n_iter": iter,
        "denoising_strength": 0.7,
        "width": 512,
        "height": 512,
        "sampler_name": "Euler a",
        "negative_prompt": "worst quality, low quality, medium quality, ugly, disgusting, gore, poor quality, low res, morbid, hideous, heinous, horror, deleted, lowres, comic, bad anatomy, bad hands, text, missing fingers, extra digit, fewer digits, cropped, watermark, blurry, pixel art, jpeg artifacts, error, painting, nsfw",
    }

    response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)

    r = response.json()
    z = 0
    newdir = f"embed_gens/{prompt}"
    if not os.path.exists(newdir):
        os.makedirs(newdir)
    for i in r['images']:
        image = Image.open(io.BytesIO(base64.b64decode(i.split(",", 1)[0])))
        z += 1
        png_payload = {
            "image": "data:image/png;base64," + i
        }
        response2 = requests.post(url=f'{url}/sdapi/v1/png-info', json=png_payload)
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("parameters", response2.json().get("info"))
        image.save(f'embed_gens/{prompt}.png', pnginfo=pnginfo)