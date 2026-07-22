import urllib.request
def check_url(url):
    req = urllib.request.Request(url, method="HEAD")
    try:
        urllib.request.urlopen(req)
        print(f"URL exists: {url}")
    except Exception as e:
        print(f"URL fails: {url}, {e}")

check_url("https://huggingface.co/akhaliq/AnimeGANv2-ONNX/resolve/main/AnimeGANv2_Hayao.onnx")
check_url("https://huggingface.co/akhaliq/AnimeGANv2-ONNX/resolve/main/animeganv2_hayao.onnx")
check_url("https://huggingface.co/vumichien/AnimeGANv2_Hayao/resolve/main/AnimeGANv2_Hayao.onnx")
check_url("https://huggingface.co/vumichien/AnimeGANv2_Hayao/resolve/main/animeganv2_hayao.onnx")
check_url("https://huggingface.co/akhaliq/AnimeGANv2/resolve/main/AnimeGANv2_Hayao.onnx")
