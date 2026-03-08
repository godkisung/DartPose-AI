# Hardware Integration Guide

다트 애플리케이션(AI Dart Coach)과 다트 보드 하드웨어 간의 통신 규격을 정의하는 문서입니다. 하드웨어/전기 엔지니어는 본 문서를 참고하여 소프트웨어 팀으로 센서 데이터를 전달해 주시기 바랍니다.

## 🔌 통신 개요

소프트웨어는 다트가 타겟에 명중(Hit)하는 순간을 정확히 감지하기 위해 하드웨어로부터 시리얼 프로토콜로 신호를 받습니다. 

- **통신 방식**: USB Serial (UART)
- **기본 Baud Rate**: `9600` bps (필요시 소프트웨어 측 `src/hardware/serial_receiver.py`에서 변경 가능)
- **엔드포인트 포트 예시**: 
  - Windows: `COM3`, `COM4`
  - Linux/Mac: `/dev/ttyACM0`, `/dev/ttyUSB0`

## 📣 데이터 패킷 포맷 (명령어)

아두이노(혹은 ESP32 등 MCU)에서 PC로 보내야 하는 문자열 규격은 다음과 같습니다. 데이터는 반드시 개행 문자(`\n`)로 끝나야 합니다.

### 1. 단순 타격 감지 (필수)
제일 기본적인 구현입니다. 다트가 보드에 박히는 진동이나 스위치 변화가 감지되면, 바로 다음 문자열을 전송해 주세요.
```text
HIT\n
```

### 2. 점수 및 위치 감지 (선택 사항)
다트 머신처럼 맞은 위치(점수)를 감지할 수 있는 시스템을 구축 중이시라면 아래와 같이 점수를 포함해 보내주시면 됩니다.
```text
HIT:20:S\n  # 20 싱글
HIT:18:T\n  # 18 트리플
HIT:BULL\n  # 불스아이
```

### 3. 리셋 신호 (선택 사항)
사용자가 3발을 다 던지고 다트를 뽑으러 갈 때 센서를 일시 중지하거나, 다음 턴을 준비하기 위한 신호입니다. (버튼 연동 추천)
```text
RESET\n
```

## 🛠️ 하드웨어 팀(아두이노) 예시 코드
```cpp
const int SENSOR_PIN = 2; // 타격 감지 센서(피에조 스피커 등) 연결 핀

void setup() {
  Serial.begin(9600);
  pinMode(SENSOR_PIN, INPUT);
}

void loop() {
  int sensorValue = digitalRead(SENSOR_PIN);
  
  if (sensorValue == HIGH) { // 타격 감지됨
    Serial.println("HIT");
    delay(100); // 디바운싱: 중복 감지 방지를 위한 딜레이
  }
}
```

## 📞 문의 및 협업
- 연동 중 시리얼 연결이 끊어지거나, 윈도우/리눅스 USB 포트 할당에 어려움이 있을 시 바로 이슈(Issue) 남겨주시면 소프트웨어 측 파서(`serial_receiver.py`)의 패킷을 점검 및 수정하겠습니다!
