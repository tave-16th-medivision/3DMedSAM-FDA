# Vast.ai GPU Server Manual (for 3DMedSAM-FDA)

이 문서는 [Vast.ai](https://vast.ai/)에서 **3DMedSAM-FDA** 프로젝트를 위한 GPU 인스턴스를 설정하고, 데이터를 세팅하여 학습/테스트를 진행하는 전체 워크플로우 가이드입니다. 

## 목차
0. [사전 준비 (계정 및 SSH Key)](#0-사전-준비-계정-및-ssh-key)
1. [인스턴스 설정 및 생성 (Custom Template)](#1-인스턴스-설정-및-생성-custom-template)
2. [서버 접속 방법 (SSH & VSCode)](#2-서버-접속-방법-ssh--vscode)
3. [초기 환경 설정 (Tmux & Nano)](#3-초기-환경-설정-tmux--nano)
4. [데이터셋 준비 (Cloud Sync & Unzip)](#4-데이터셋-준비-cloud-sync--unzip)
5. [학습 및 테스트 실행](#5-학습-및-테스트-실행)
6. [인스턴스 관리 (주의사항)](#6-인스턴스-관리-주의사항)

---

## 0. 사전 준비 (계정 및 SSH Key)

### 0.1 계정 생성 및 크레딧 충전
1. [Vast.ai Console](https://cloud.vast.ai/login/)에 접속하여 회원가입을 진행합니다.
2. **Billing** 메뉴로 이동하여 크레딧을 충전합니다 (신용카드 또는 암호화폐).
   > **참고:** Vast.ai는 선불 충전 방식입니다. 잔액이 $0 미만이 되면 인스턴스가 강제로 중지될 수 있습니다.

### 0.2 SSH Public Key 등록 (필수)
비밀번호 없이 안전하게 접속하기 위해 SSH Key를 등록해야 합니다.

1. 로컬 터미널(내 컴퓨터)에서 키 생성 (이미 있다면 생략):
   ```bash
   ssh-keygen -t rsa
   ```
   (모든 질문에 Enter를 누르면 기본 경로 `~/.ssh/id_rsa.pub`에 생성됩니다.)

2. Public Key 내용 복사:
   ```bash
   cat ~/.ssh/id_rsa.pub
   ```
3. Vast.ai 콘솔의 **Account** > **SSH Keys** 섹션에 복사한 내용을 붙여넣고 `ADD`를 클릭합니다.

---

## 1. 인스턴스 설정 및 생성 (Custom Template)

### 1.1 템플릿 생성 (최초 1회)
프로젝트 환경에 맞는 커스텀 템플릿을 생성해야 합니다.

1. Vast.ai 콘솔의 **Templates** 탭 > **New** 버튼 클릭.
2. **Config** 설정에서 아래 내용을 입력합니다:
   * **Template Name:** `3DMedSAM-cu113`
   * **Docker Repository (Image Path:Tag):** `shg0592/med-sam-cu113:cu113`
   * 나머지는 기본값 유지 후 **Create** 클릭.

### 1.2 인스턴스 검색 및 대여
1. **Search** 탭에서 **Change Template**을 눌러 위에서 만든 `3DMedSAM-cu113`을 선택합니다.
2. 필터 설정:
   * **Container Size:** `200GB` (데이터셋 용량을 고려하여 넉넉하게 설정)
   * **GPU Model:** `A40` 필터링
3. 목록 중 가격이 가장 저렴하고 신뢰도(Reliability)가 높은 매물을 선택하여 **RENT** 클릭.

---

## 2. 서버 접속 방법 (SSH & VSCode)

**Instances** 탭에서 상태가 `Running`이 되면 접속할 수 있습니다.

### 2.1 SSH Key 등록
1. 로컬 터미널에서 Public Key 확인: `cat ~/.ssh/id_rsa.pub`
2. 인스턴스 카드의 열쇠 모양 아이콘 클릭 > **New SSH Key** 입력란에 붙여넣기 > **Add SSH KEY**.

### 2.2 터미널에서 SSH 접속 
터미널에서 ssh 명령어를 통해 곧바로 서버에 원격 접속이 가능합니다. 

1. 접속 명령어 확인: 인스턴스 카드의 열쇠 모양 아이콘 클릭 >  **Direct ssh connect**에서 명령어 확인. 
2. 터미널에서 명령어 붙여넣기 
    ```bash
    ssh -p 64704 root@116.101.122.173 -L 8080:localhost:8080 
    ```

### 2.3 VSCode Remote SSH 접속 
VSCode를 사용하면 코드 수정과 터미널 작업을 동시에 할 수 있어 효율적입니다.

1. VSCode 확장 프로그램 설치: `Remote - SSH`
2. **Remote Explorer** > 설정(톱니바퀴) > `config` 파일 선택.
3. 접속 정보 입력 (Vast.ai에서 제공하는 IP와 Port 확인):
   ```ssh
   # 예시: ssh -p 64704 root@116.101.122.173
   Host vastai_medivision
       HostName 116.101.122.173
       User root
       Port 64704
       IdentityFile ~/.ssh/id_rsa
    ```

* `IdentityFile`에는 로컬의 Private Key 경로를 입력합니다.

---

## 3. 초기 환경 설정 (Tmux 터미널 내 스크롤 설정)

SSH 연결이 끊겨도 작업이 유지되도록 `tmux` 터미널 접속 방식을 권장합니다.

이때 Tmux 터미널 접속 시 스크롤이 되지 않는 문제를 아래 과정을 통해 해결합니다. 


### 3.1 Nano 에디터 설치 및 Tmux 설정

터미널 접속 후 아래 명령어를 순서대로 입력합니다.

```bash
# 1. Nano 에디터 설치
apt-get install nano -y

# 2. Tmux 설정 파일 생성 (마우스 스크롤 및 vi 키 바인딩 활성화)
nano ~/.tmux.conf
```

`~/.tmux.conf` 파일 내용:

```conf
# 마우스 스크롤 및 복사 모드 활성화
set -g mouse on

# vi 키 바인딩 사용 (hjkl로 탐색)
setw -g mode-keys vi
# 복사 모드에서 선택 시작: v, 선택 완료 및 복사: y
bind -T copy-mode-vi v send -X begin-selection
bind -T copy-mode-vi y send -X copy-selection-and-cancel
```

설정 적용:
```bash
tmux source-file ~/.tmux.conf
```

---

## 4. 데이터셋 준비 (Cloud Sync & Unzip)

대용량 데이터는 **AWS S3 Cloud Connection** 기능을 사용하여 전송합니다.

### 4.1 Cloud Connection 설정

1. Vast.ai 콘솔 **Account** > **Cloud Connection** > **Connect Amazon S3** 클릭.
2. Access Key 및 Secret Key 입력 (팀 내 공유된 키 사용).

### 4.2 데이터 전송 (S3 -> Instance)

인스턴스 카드의 **구름 모양 아이콘** 클릭 > **MIGRATING FROM CLOUD TO INSTANCE** 탭 확인.
아래 경로를 입력하고 **SYNC** 버튼을 누릅니다.

| 데이터셋 및 가중치 | Item Location in S3 | Path to Destination |
| --- | --- | --- |
| **KiTS** | `medivision/kist_update.zip` | `/workspace/` |
| **LiTS** | `medivision/Task01_LITS17.zip` | `/workspace/` |
| **Pancreas** | `medivision/Task03_pancreas.zip` | `/workspace/` |
| **Colon** | `medivision/Task10_Colon.zip` | `/workspace/` |
| **Weights** | `medivision/sam_vit_b_01ec64.pth` | `/workspace/` |

### 4.3 압축 해제 및 파일 정리

전송 완료 후 터미널에서 다음 명령어를 실행하여 디렉토리 구조를 맞춥니다.

```bash
# 0. Unzip 설치
apt-get update && apt-get install unzip -y

# 1. 데이터셋 압축 해제 
# (데이터셋 별로 경로 압축파일명 바꿔서 명령어 실행)
mkdir -p /workspace/dataset/
unzip /workspace/kist_update.zip -d /workspace/dataset/

# 2. Pre-trained Weight 이동
mkdir -p /workspace/3DSAM-adapter/3DSAM-adapter/ckpt/
mv /workspace/sam_vit_b_01ec64.pth /workspace/3DSAM-adapter/3DSAM-adapter/ckpt/
```

---

## 5. 학습 및 테스트 실행

기본 작업 경로는 `/workspace/3DSAM-adapter/3DSAM-adapter/` 입니다.

### 5.1 학습 (Training)

```bash
# 1. 가중치 저장 폴더 생성
mkdir -p /workspace/weights/

# 2. 경로 이동 및 학습 실행
cd /workspace/3DSAM-adapter/3DSAM-adapter
python train.py --data kits --snapshot_path "/workspace/weights/" --data_prefix "/workspace/dataset/kist_update/data/" --max_epoch 200
```

> **Tip:** 학습 시작 시 `libcuda.so` 관련 에러 발생 시 해결법:
> ```bash
> find /usr -name "libcuda.so*" 2>/dev/null
> # 찾은 경로를 이용하여 심볼릭 링크 생성 (예: libcuda.so.1 -> libcuda.so)
> ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so
> ```
> 

### 5.2 테스트 (Testing)

테스트 로그를 파일로 저장하기 위해 `-u` 옵션과 리다이렉션(`>`)을 사용합니다.

**KiTS 데이터셋 예시:**

```bash
# Best Checkpoint + 1 Point Prompt
python -u test.py --data kits \
  --snapshot_path "/workspace/weights/" \
  --data_prefix "/workspace/dataset/kist_update/data/" \
  --num_prompts 1 \
  --checkpoint best > /workspace/weights/kits/test_kits_best_1pt.log 2>&1
```

**주요 옵션:**

* `--num_prompts`: 점 프롬프트 개수 설정 (1, 3, 10 등)
* `--checkpoint`: `best` (검증 최고 성능) 또는 생략 시 `last` (마지막 에폭)

---

## 6. 인스턴스 관리 (주의사항)

* **STOP:** 인스턴스를 중지합니다. GPU 비용은 안 나가지만, **스토리지 비용(약 200GB분)**은 계속 청구됩니다.
* **DESTROY:** 인스턴스를 완전히 삭제합니다. **모든 데이터가 사라지므로**, 필요한 가중치(`weights` 폴더)나 로그 파일은 반드시 로컬로 다운로드(`scp` 사용) 후 삭제하세요.

```bash
# (로컬 터미널에서) 학습 결과 다운로드 예시
scp -P [PORT] -r root@[IP]:/workspace/weights/ ./my_local_backup/

```

