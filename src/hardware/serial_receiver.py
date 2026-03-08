"""다트 보드 하드웨어 시리얼 통신 모듈.

Arduino 기반 타격 센서와의 USB Serial 통신을 담당합니다.
키보드 시뮬레이터 모드를 지원하여 하드웨어 없이도 테스트 가능합니다.
"""

import time
import threading
from typing import Callable

from src.config import (
    SERIAL_BAUDRATE,
    SERIAL_DEFAULT_PORT,
    SERIAL_TIMEOUT,
    SERIAL_RECONNECT_DELAY,
)


class HitEvent:
    """타격 이벤트 데이터."""

    def __init__(self, timestamp_ms: float, score: str = "", zone: str = ""):
        self.timestamp_ms = timestamp_ms
        self.score = score       # "20", "BULL" 등
        self.zone = zone         # "S"(싱글), "D"(더블), "T"(트리플)
        self.pc_timestamp = time.time()

    def __repr__(self):
        if self.score:
            return f"HIT:{self.score}:{self.zone} @ {self.timestamp_ms}ms"
        return f"HIT @ {self.timestamp_ms:.0f}ms"


class ArduinoReceiver:
    """Arduino 시리얼 수신기.

    타격 이벤트를 비동기적으로 수신하고 콜백으로 전달합니다.
    """

    def __init__(
        self,
        port: str = SERIAL_DEFAULT_PORT,
        baudrate: int = SERIAL_BAUDRATE,
        on_hit: Callable[[HitEvent], None] | None = None,
        on_reset: Callable[[], None] | None = None,
    ):
        self.port = port
        self.baudrate = baudrate
        self.on_hit = on_hit
        self.on_reset = on_reset
        self._ser = None
        self._running = False
        self._thread = None

    def connect(self) -> bool:
        """시리얼 포트에 연결합니다."""
        try:
            import serial
            self._ser = serial.Serial(self.port, self.baudrate, timeout=SERIAL_TIMEOUT)
            print(f"✓ Arduino 연결: {self.port} @ {self.baudrate}")
            return True
        except Exception as e:
            print(f"⚠ Arduino 연결 실패: {e}")
            return False

    def start_listening(self):
        """백그라운드 스레드에서 이벤트 수신을 시작합니다."""
        if not self._ser:
            print("⚠ 시리얼 미연결 — start_listening 무시")
            return

        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        print("🔊 타격 이벤트 대기 중...")

    def stop(self):
        """수신을 중지합니다."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._ser:
            self._ser.close()
        print("🔇 시리얼 수신 종료")

    def _listen_loop(self):
        """이벤트 수신 루프."""
        while self._running:
            try:
                if self._ser and self._ser.in_waiting > 0:
                    line = self._ser.readline().decode('utf-8', errors='ignore').strip()
                    self._parse_message(line)
            except Exception as e:
                print(f"⚠ 시리얼 읽기 오류: {e}")
                time.sleep(SERIAL_RECONNECT_DELAY)
                self._try_reconnect()

    def _try_reconnect(self):
        """연결이 끊어졌을 때 재연결을 시도합니다."""
        try:
            if self._ser:
                self._ser.close()
            import serial
            self._ser = serial.Serial(self.port, self.baudrate, timeout=SERIAL_TIMEOUT)
            print(f"✓ Arduino 재연결 성공: {self.port}")
        except Exception:
            pass

    def _parse_message(self, line: str):
        """수신된 메시지를 파싱합니다.

        지원 포맷:
          HIT
          HIT,timestamp
          HIT:score:zone
          RESET
        """
        if not line:
            return

        if line == "RESET":
            print("🔄 RESET 수신")
            if self.on_reset:
                self.on_reset()
            return

        if line.startswith("HIT"):
            event = self._parse_hit(line)
            if self.on_hit:
                self.on_hit(event)

    @staticmethod
    def _parse_hit(line: str) -> HitEvent:
        """HIT 메시지를 파싱합니다."""
        ts = time.time() * 1000  # fallback 타임스탬프

        if "," in line:
            # HIT,timestamp
            parts = line.split(",")
            if len(parts) >= 2:
                try:
                    ts = float(parts[1])
                except ValueError:
                    pass
            return HitEvent(timestamp_ms=ts)

        if ":" in line:
            # HIT:score:zone  또는  HIT:BULL
            parts = line.split(":")
            score = parts[1] if len(parts) > 1 else ""
            zone = parts[2] if len(parts) > 2 else ""
            return HitEvent(timestamp_ms=ts, score=score, zone=zone)

        return HitEvent(timestamp_ms=ts)


class KeyboardSimulator:
    """키보드 기반 하드웨어 시뮬레이터.

    Enter 키를 누르면 HIT 이벤트를 생성합니다.
    하드웨어 없이 소프트웨어 테스트/데모가 가능합니다.
    """

    def __init__(
        self,
        on_hit: Callable[[HitEvent], None] | None = None,
        on_reset: Callable[[], None] | None = None,
    ):
        self.on_hit = on_hit
        self.on_reset = on_reset
        self._running = False
        self._thread = None

    def start(self):
        """키보드 입력 수신을 시작합니다."""
        self._running = True
        self._thread = threading.Thread(target=self._input_loop, daemon=True)
        self._thread.start()
        print("⌨ 시뮬레이터 모드: Enter=HIT, r=RESET, q=QUIT")

    def stop(self):
        self._running = False

    def _input_loop(self):
        while self._running:
            try:
                user_input = input().strip().lower()
                if user_input == "" or user_input == "h":
                    event = HitEvent(timestamp_ms=time.time() * 1000)
                    print(f"  → 시뮬레이션 {event}")
                    if self.on_hit:
                        self.on_hit(event)
                elif user_input == "r":
                    print("  → 시뮬레이션 RESET")
                    if self.on_reset:
                        self.on_reset()
                elif user_input == "q":
                    self._running = False
                    break
            except EOFError:
                break
