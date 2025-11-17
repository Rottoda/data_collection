import cv2

index = 0
while True:
    # 해당 인덱스의 카메라를 열려고 시도
    cap = cv2.VideoCapture(index, cv2.CAP_VFW)
    
    if not cap.isOpened():
        print(f"카메라 인덱스 {index}를 열 수 없습니다. 테스트를 종료합니다.")
        break
    
    print(f"카메라 인덱스 {index} 테스트 중... (성공!)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 화면에 현재 테스트 중인 인덱스 번호를 표시
        cv2.putText(frame, f"Index: {index}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Camera Test', frame)
        
        # 'q' 키를 누르면 다음 인덱스로 넘어감
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    index += 1

print("모든 사용 가능한 카메라를 확인했습니다.")