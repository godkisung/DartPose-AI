# Hardware Integration Guide

다트 애플리케이션(AI Dart Coach)과 다트 보드 하드웨어 간의 통신 규격을 정의하는 문서입니다.
하드웨어/전기 엔지니어는 본 문서를 참고하여 소프트웨어 팀으로 센서 데이터를 전달해 주시기 바랍니다.

## 🔌 통신 개요

소프트웨어는 다트가 타겟에 명중(Hit)하는 순간을 정확히 감지하기 위해 하드웨어로부터 시리얼 프로토콜로 신호를 받습니다.

- **통신 방식**: USB Serial (UART)
- **기본 Baud Rate**: **`115200`** bps
- **통신 방향**: 양방향 (Arduino → PC 필수, PC → Arduino 선택)
- **엔드포인트 포트 예시**: 
  - Windows: `COM3`, `COM4`
  - Linux/Mac: `/dev/ttyACM0`, `/dev/ttyUSB0`

### 소프트웨어 측 연동 파일

| 파일 | 역할 |
|------|------|
| `src/hardware/serial_receiver.py` | 시리얼 수신 및 이벤트 콜백 처리 |
| `src/config.py` | Baud Rate, 포트, 타임아웃 등 설정값 |

---

## 📣 데이터 패킷 포맷 (Arduino → PC)

아두이노(혹은 ESP32 등 MCU)에서 PC로 보내야 하는 문자열 규격입니다.
**데이터는 반드시 개행 문자(`\n`)로 끝나야 합니다.**

### 1. 단순 타격 감지 (필수 — 최소 구현)

다트가 보드에 박히는 진동이나 스위치 변화가 감지되면, 바로 다음 문자열을 전송:
```text
HIT\n
```

### 2. 타격 + MCU 타임스탬프 (권장)

MCU 내부 타이머의 밀리초 단위 타임스탬프를 포함하면, PC 측에서 프레임 동기화 정밀도가 향상됩니다:
```text
HIT,102450\n
```
> 포맷: `이벤트타입,MCU_Timestamp_ms`

### 3. 점수 및 위치 감지 (선택 사항)

다트 머신처럼 맞은 위치(점수)를 감지할 수 있는 경우:
```text
HIT:20:S\n   # 20 싱글
HIT:18:T\n   # 18 트리플
HIT:20:D\n   # 20 더블
HIT:BULL\n   # 불스아이
```

### 4. 리셋 신호 (선택 사항)

3발 투구 후 다트를 뽑으러 갈 때 센서를 일시 중지하거나, 다음 턴을 준비하기 위한 신호:
```text
RESET\n
```
> 물리 버튼을 연동하여 사용자가 직접 누르는 방식을 추천합니다.

---

## 🛠️ 하드웨어 팀(아두이노) 예시 코드

### 기본 구현 (단순 HIT)
```cpp
const int SENSOR_PIN = 2;                 // 타격 감지 센서 연결 핀
const int DEBOUNCE_DELAY_MS = 500;        // 디바운싱 딜레이 (ms)
unsigned long lastHitTime = 0;

void setup() {
  Serial.begin(115200);                   // ⚠ 반드시 115200으로 설정
  pinMode(SENSOR_PIN, INPUT);
}

void loop() {
  int sensorValue = digitalRead(SENSOR_PIN);
  unsigned long now = millis();

  if (sensorValue == HIGH && (now - lastHitTime) > DEBOUNCE_DELAY_MS) {
    lastHitTime = now;
    // 타임스탬프 포함 권장
    Serial.print("HIT,");
    Serial.println(now);
  }
}
```

### 확장 구현 (HIT + RESET 버튼)
```cpp
const int SENSOR_PIN = 2;
const int RESET_BUTTON_PIN = 3;
const int DEBOUNCE_DELAY_MS = 500;
unsigned long lastHitTime = 0;

void setup() {
  Serial.begin(115200);
  pinMode(SENSOR_PIN, INPUT);
  pinMode(RESET_BUTTON_PIN, INPUT_PULLUP);  // 내부 풀업 사용
}

void loop() {
  unsigned long now = millis();

  // 타격 감지
  if (digitalRead(SENSOR_PIN) == HIGH && (now - lastHitTime) > DEBOUNCE_DELAY_MS) {
    lastHitTime = now;
    Serial.print("HIT,");
    Serial.println(now);
  }

  // 리셋 버튼 (LOW가 눌림)
  if (digitalRead(RESET_BUTTON_PIN) == LOW) {
    Serial.println("RESET");
    delay(300);  // 리셋 버튼 디바운싱
  }
}
```

---

## 🔧 전기 엔지니어 확장 과제

소프트웨어 시스템은 아래 확장 기능을 지원할 준비가 되어 있습니다.
하드웨어 측에서 구현하면 즉시 연동됩니다:

1. **다중 센서 TDOA 로직**: 3~4개 Piezo 센서의 신호 도달 시간차(TDOA)를 MCU에서 계산하여 1차 좌표를 추정하고, `HIT:x:y` 포맷으로 전송
2. **하드웨어 디바운싱**: 소프트웨어 `delay()` 대신 RC Filter + Schmitt Trigger 회로 구현으로 노이즈 감쇠
3. **LED 피드백 수신 (PC → Arduino)**: 분석 결과를 아두이노로 역전송하여 LED/부저로 피드백 표시 (양방향 통신)

---

## 📞 문의 및 협업

- 연동 중 시리얼 연결이 끊어지거나, 윈도우/리눅스 USB 포트 할당에 어려움이 있을 시 바로 이슈(Issue)를 남겨주시면 소프트웨어 측에서 점검하겠습니다.
- **하드웨어 없이 테스트**: `uv run python src/main.py --mode live --simulate` 명령으로 키보드 시뮬레이터를 사용할 수 있습니다 (Enter=HIT, r=RESET).
