import cv2

from app.main import predict


def main() -> None:
    image_path = 'samples/KakaoTalk_20230320_163338129.jpg'
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print(predict(image))


if __name__ == '__main__':
    main()
