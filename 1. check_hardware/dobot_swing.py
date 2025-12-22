import sys
import os
import threading
import numpy as np
from time import sleep, time

# 제공된 코드의 Dobot API를 사용하기 위한 import
# 'TCP_IP_4Axis_Python' 폴더가 이 스크립트와 동일한 상위 폴더에 있다고 가정합니다.
# 경로가 다른 경우, 이 부분을 적절히 수정해야 합니다.
try:
    script_base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(script_base_dir)
    from TCP_IP_4Axis_Python.dobot_api import DobotApiDashboard, DobotApi, DobotApiMove, MyType
except ImportError:
    print("="*60)
    print("ERROR: Dobot API 라이브러리를 찾을 수 없습니다.")
    print("이 스크립트가 'TCP_IP_4Axis_Python' 폴더와 올바른 상대 경로에 있는지 확인하세요.")
    print("="*60)
    sys.exit(1)


# ==================================================================
#                           스윙 동작 설정
# ==================================================================
CONFIG = {
    # === 로봇 연결 설정 ===
    "robot_ip": "192.168.1.6",  # 제어할 로봇의 IP 주소

    # === 스윙 동작 파라미터 ===
    "center_x": 340.0,              # 스윙 동작의 중심 X좌표 (mm)
    "center_y": 7,                # 스윙 동작의 중심 Y좌표 (mm)
    "swing_height_z": -130.0,       # 스윙 동작을 수행할 고정 높이 Z좌표 (mm)
    "orientation_r": 352.9,         # 스윙 중 유지할 로봇의 회전 각도 (R)
    
    "swing_distance_mm": 30.0,      # 중심으로부터 좌/우로 움직일 거리 (mm). 총 스윙 폭은 이 값의 2배.
    "robot_speed": 50,              # 로봇 이동 속도 (1~100)
    "num_swings": 5,               # 총 왕복 스윙 횟수
}

# ==================================================================
#                          로봇 제어 클래스
# ==================================================================
class RobotSwingController:
    """로봇 스윙 동작을 위한 연결 및 제어 클래스"""
    def __init__(self, config):
        self.config = config
        self.dashboard = None
        self.move = None
        self.feed = None
        self.current_actual = None
        self.lock = threading.Lock()

    def connect(self):
        """로봇에 연결하고 활성화합니다."""
        print("[Controller] 로봇 연결 시작...")
        ip = self.config['robot_ip']
        self.dashboard = DobotApiDashboard(ip, 29999)
        self.move = DobotApiMove(ip, 30003)
        self.feed = DobotApi(ip, 30004)
        print(" > 로봇 API 연결 성공.")
        
        self.dashboard.EnableRobot()
        print(" > 로봇 활성화 완료.")
        
        # 로봇 상태 피드백을 위한 쓰레드 시작
        threading.Thread(target=self._get_feed, daemon=True).start()
        sleep(1) # 활성화 및 피드백 시작 대기
        print("[Controller] 로봇 연결 및 준비 완료.")

    def disconnect(self):
        """로봇을 비활성화하고 연결을 해제합니다."""
        print("\n[Controller] 로봇 연결 해제 시작...")
        if self.dashboard:
            self.dashboard.DisableRobot()
            print(" > 로봇 비활성화 완료.")
        print("[Controller] 로봇 연결 해제 완료.")

    def _get_feed(self):
        """(참조 코드 기반) 로봇의 현재 상태를 실시간으로 수신합니다."""
        hasRead = 0
        while True:
            data = bytes()
            while hasRead < 1440:
                temp = self.feed.socket_dobot.recv(1440 - hasRead)
                if len(temp) > 0:
                    hasRead += len(temp)
                    data += temp
            hasRead = 0
            feedInfo = np.frombuffer(data, dtype=MyType)
            if hex((feedInfo['test_value'][0])) == '0x123456789abcdef':
                with self.lock:
                    # tool_vector_actual: [x, y, z, r, j5, j6]
                    self.current_actual = feedInfo["tool_vector_actual"][0]
            sleep(0.001)

    def wait_for_arrival(self, target_point):
        """(참조 코드 기반) 로봇이 목표 지점에 도착할 때까지 대기합니다."""
        while True:
            if self.current_actual is not None:
                # X, Y, Z, R 좌표만 비교 (오차 허용범위 1mm/1도)
                if np.all(np.abs(np.array(self.current_actual[:4]) - np.array(target_point[:4])) < 1.0):
                    break
            sleep(0.001)

    def move_to_and_wait(self, x, y, z, r):
        """지정된 좌표로 이동하고 도착할 때까지 기다립니다."""
        target = [x, y, z, r]
        print(f" > 이동 목표: X={x:.2f}, Y={y:.2f}, Z={z:.2f}, R={r:.2f}")
        self.move.MovL(x, y, z, r, 1, 1, self.config['robot_speed'])
        self.wait_for_arrival(target)
        print(" > 도착 완료.")

# ==================================================================
#                          메인 실행 함수
# ==================================================================
def main():
    """스윙 동작을 실행하는 메인 함수"""
    controller = RobotSwingController(CONFIG)
    
    try:
        # 1. 로봇 연결
        controller.connect()

        # 2. 스윙 좌표 계산
        cx, cy, cz, cr = CONFIG['center_x'], CONFIG['center_y'], CONFIG['swing_height_z'], CONFIG['orientation_r']
        dist = CONFIG['swing_distance_mm']
        
        left_y = cy - dist
        right_y = cy + dist
        
        print("\n" + "="*40)
        print("스윙 동작을 시작합니다.")
        print(f"중심: (X: {cx}, Y: {cy}, Z: {cz})")
        print(f"왼쪽 목표 Y: {left_y:.2f} | 오른쪽 목표 Y: {right_y:.2f}")
        print(f"속도: {CONFIG['robot_speed']}, 반복 횟수: {CONFIG['num_swings']}")
        print("="*40 + "\n")

        # 3. 안전한 시작 위치(중앙)로 먼저 이동
        print("--- 시작 위치로 이동 ---")
        controller.move_to_and_wait(cx, cy, cz, cr)
        sleep(1)

        # 4. 설정된 횟수만큼 스윙 반복
        for i in range(CONFIG['num_swings']):
            print(f"\n--- 스윙 {i + 1}/{CONFIG['num_swings']} ---")
            
            # 오른쪽으로 이동
            controller.move_to_and_wait(cx, right_y, cz, cr)
            sleep(0.2)
            
            # 왼쪽으로 이동
            controller.move_to_and_wait(cx, left_y, cz, cr)
            sleep(0.2)

        # 5. 완료 후 중앙으로 복귀
        print("\n--- 스윙 동작 완료. 중앙으로 복귀 ---")
        controller.move_to_and_wait(cx, cy, cz, cr)
        print("\n[SUCCESS] 모든 동작을 성공적으로 완료했습니다.")

    except (KeyboardInterrupt, SystemExit):
        print("\n[STOP] 사용자에 의해 프로그램이 중단되었습니다.")
    except Exception as e:
        print(f"\n[FATAL ERROR] 예상치 못한 오류로 프로그램을 중단합니다: {e}")
    finally:
        # 6. 로봇 연결 해제 (오류 발생 시에도 실행)
        controller.disconnect()
        print("\n프로그램이 완전히 종료되었습니다.")

if __name__ == "__main__":
    main()