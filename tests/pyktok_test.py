import pyktok as pyk


def test_pyktok():
    data = pyk.alt_get_tiktok_json('https://www.tiktokv.com/share/video/7523174241471089942/')
    print(data)


def test_pyktok_save_video():
    pyk.save_tiktok('https://www.tiktokv.com/share/video/6935064737638141186/')
