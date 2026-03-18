import os
import json 
import csv

folder_path = './'
results = []
output_csv = 'result.csv'

for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            throws_cnt = data.get('total_throws_detected')
            if throws_cnt == 3:
                status = '인식 잘 되는 자세임'
                print(f"좋은 자세 자세 특징을 찾아보자 File: {filename}, {throws_cnt}")
            else:
                status = '인식 안 되는 자세임 큰 차이 없어보이는데,,, 왜일까'
                print(f"안 좋은 자세 특징을 찾아보자 File: {filename}, {throws_cnt}")
                
            results.append({
                        'filename': filename,
                        'throws_count': throws_cnt,
                        'status': status
                    })

# --- 루프가 끝난 후 CSV 저장 ---
keys = results[0].keys() if results else []
with open(output_csv, 'w', newline='', encoding='utf-8-sig') as output_file:
    dict_writer = csv.DictWriter(output_file, fieldnames=keys)
    dict_writer.writeheader()  # 헤더(컬럼명) 쓰기
    dict_writer.writerows(results)  # 데이터 쓰기

print(f"\n💾 저장이 완료되었습니다: {output_csv}")