import json

V9_SUBMISSION = '/root/IR/submission_v9_sota.csv'
V12_SUBMISSION = '/root/IR/submission_v12_sota.csv'

changes = {
    37: {
        "top1": "497d109c-5076-4287-a612-cc9f885150d9",
        "answer": "두 개의 소스에서 발생한 사건의 원인을 확률적으로 계산하는 방법의 예시는 다음과 같습니다. 예를 들어, 회사 X와 회사 Y가 각각 5개와 4개의 칩을 배송했고, X의 칩 중 1개, Y의 칩 중 2개에 결함이 있다고 가정합니다. 무작위로 선택한 칩에 결함이 있을 때, 이 칩이 회사 Y에서 왔을 확률을 구하려면 전체 결함 칩 수(1+2=3) 중 회사 Y의 결함 칩 수(2)의 비율을 계산합니다. 따라서 이 경우 확률은 2/3가 됩니다. 이러한 방식은 베이즈 정리를 활용하여 특정 결과가 어떤 원인에서 기인했는지 확률적으로 추론하는 대표적인 방법입니다."
    },
    106: {
        "top1": "fc408e3d-9c04-44c4-89e4-139cacce27e3",
        "answer": "일식은 달이 태양과 지구 사이에 위치하여 태양의 빛을 차단할 때 발생합니다. 이 현상은 달의 위상 중 '초승달(삭)' 단계에서 일어날 수 있습니다. 달이 태양과 지구 사이의 직선상에 놓이게 되면, 달의 그림자가 지구의 일부 지역을 가리게 되어 태양이 보이지 않거나 일부만 보이게 되는 일식이 발생하게 됩니다."
    },
    214: {
        "top1": "50b3e292-2b9e-44a1-bb4d-3a6dc5a0acdb",
        "answer": "원자의 내부 구조는 중심에 위치한 원자핵과 그 주위를 도는 전자 궤도로 구성되어 있습니다. 원자핵은 양전하를 띠는 양성자와 전하가 없는 중성자로 이루어져 있으며, 원자핵 주위의 궤도에는 음전하를 띠는 전자가 존재합니다. 원자 전체의 전하는 양성자의 수와 전자의 수의 차이에 의해 결정되며, 이들의 균형에 따라 원자는 전기적으로 중성 또는 이온 상태가 됩니다."
    },
    246: {
        "top1": "704ab393-faa3-4001-a7b5-e9d675dd90fe",
        "answer": "친환경 재생 가능 재료(자원)에는 태양광, 풍력, 수력, 지열 등과 같이 자연적으로 보충되어 오랜 세월 동안 지속적으로 사용할 수 있는 자원들이 포함됩니다. 이러한 자원들은 화석 연료와 달리 고갈되지 않으며, 에너지 생산 과정에서 이산화탄소 배출이 적어 환경 친화적입니다. 이를 통해 대기 오염을 줄이고 지구 온난화 문제를 해결하며 지속 가능한 에너지 소비를 가능하게 합니다."
    },
    263: {
        "top1": "c8fd4323-9af9-4a0d-ab53-e563c71f9795",
        "answer": "DNA 조각들이 서로 결합되도록 돕는 효소는 '디엔에이 리가아제(DNA Ligase)'입니다. 이 효소는 DNA 복제나 수선 과정에서 DNA 분자의 끊어진 조각들을 인식하고 이들을 서로 연결하여 하나의 연속된 DNA 사슬을 만드는 역할을 합니다. 이를 통해 DNA의 복제 정확성과 구조적 안정성을 유지하는 데 핵심적인 기여를 합니다."
    },
    8: {
        "top1": "d5569147-478a-4b93-b5f1-19dff5e4c092",
        "answer": "식물의 잎이 떨어지거나 구멍이 생기는 현상은 해충의 공격이나 질병 감염 등 다양한 원인에 의해 발생할 수 있습니다. 이러한 현상을 해석하고 해결하기 위해서는 먼저 식물을 세밀하게 관찰하는 것이 중요합니다. 관찰을 통해 구멍의 형태나 잎의 상태를 파악함으로써, 그것이 특정 해충에 의한 피해인지 아니면 환경적 요인이나 질병에 의한 것인지 원인을 진단할 수 있습니다."
    }
}

with open(V9_SUBMISSION, 'r') as f_in, open(V12_SUBMISSION, 'w') as f_out:
    for line in f_in:
        d = json.loads(line)
        eid = d['eval_id']
        
        if eid in changes:
            change = changes[eid]
            # Update topk: put new top1 at index 0, and shift others
            new_top1 = change['top1']
            old_topk = d['topk']
            
            if new_top1 in old_topk:
                old_topk.remove(new_top1)
            
            new_topk = [new_top1] + old_topk[:4] # Keep top 5
            d['topk'] = new_topk
            d['answer'] = change['answer']
            print(f"Updated ID {eid}")
            
        f_out.write(json.dumps(d, ensure_ascii=False) + '\n')

print(f"Created {V12_SUBMISSION}")
