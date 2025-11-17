import cv2

index = 0
while True:
    cap = cv2.VideoCapture(index, cv2.CAP_VFW)
    
    if not cap.isOpened():
        print(f"카메라 인덱스 {index}를 열 수 없습니다. 테스트를 종료합니다.")
        break
    
    print(f"카메라 인덱스 {index} 테스트 중... (성공!)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.putText(frame, f"Index: {index}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Camera Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    index += 1

print("모든 사용 가능한 카메라를 확인했습니다.")