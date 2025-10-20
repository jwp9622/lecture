# 2025.5.12 git 기본 명령어 정리

## git 환경설정
git config --list ==> 환경설정 목록 보여주기   
git config --global user.name = "홍길동" ==> 이름 설정   
git config --global.user.email = "aaa@naver.com ==> 이메일 설정   


## 리눅스 명령어
rm <파일명> ==> 파일 삭제   
rm -r <파일명> ==> 파일이나 디렉토리를 삭제한다.   
pwd ==> 현재 경로로 표시한다.   
ls ==> 현재 디렉토리 안의 내용을 표시한다.   
ls -l ==> 현재 디렉토리 안의 숨긴 파일과 숨긴 디렉토리도 표시한다.   
ls -a ==> 현재 디렉토리안의 파일와 폴더 상세 정보까지 표시한다.   
ls -al ==> 숨길파일 상세정보 표시   
cd .. ==> 부모 디렉토리로 이동한다.   
cd ~ ==> 홈 디렉토리로 이동한다.   
clear ==> 화면을 깨끗하게 지운다.   
cd 경로 ==> 하위 디렉토리로 이동한다.   
mkdir ==> 새 디렉토리를 만든다.   
exit ==> 터미널 창을 종료한다.   
touch new.txt ==> new.txt 생성   
echo "aaa" >> new.txt ==>  aaa 내용을 넣은 new.txt 생성, new.txt가 없으면 생성하고 있으면 덮어 쒸운다   
echo "# 제목" > README.md ==> README.md 파일 생성   
.gitignore ==> 파일 생성 후 제외할 파일명 작성, 정규식 사용   


## git 기본 명령어
git init ==> 깃 초기화   
git add . ==> 깃 스테이징에 추가   
git commit -m "메시지명" ==> 커밋으로 옮기기   
git commit -am "메시지명" ==> 커밋으로 옮기기   
  - m : 인라인 형식으로 커미 메시지 작성,기본 명령어   
  - a : add,commit 한번에 수행(단, 한번도 add되지 않은 파일은 add로 따로 작업해야함)   
git status ==> 상태확인   
git log ==> 커밋한 로고 전체보기   
git log --oneline --branches --graph ==> 커밋한 목록 그래프로 보기   
git log --oneline --all --graph ==> 커밋한 전체 보기   
git log --oneline -n 5 ==> 1줄로 5개까지 보여줌.   
  - git 로그 빠져나갈때 q 누름   
git log main..apple ==> 메인에는 없고 apple에 있는것만 보여주기   
git log apple..main ==>  apple에는 없고 메인에 있는것만 보여주기   
   
git diff ==> 줄   
   
git reset ==> 커밋한 내용 삭제   
--soft: 커밋 취소, Staging 상태 유지(add)   
--mixed: 커밋 취소, Staging 취소, local은 변경 상태로 유지 (옵션설정 없을시 default)   
--hard: 커밋취소, Staging 취소, local 변경 상태 취소   
HEAD 옵션 위의 3가지 옵션뒤에 사용한다. (--soft HEAD^)   
HEAD^ : 최신 커밋 취소   
HEAD~(수량) : 수량에 숫자를 적으면 해당 숫자만큼 최근 커밋부터 해당 숫자까지 커밋 취소.   
  ex) git reset HEAD^ ==> 최상위 한개 커밋 삭제   
  ex) git reset --soft HEAD~1 ==> 커밋만 취소하되 최상위에서 1개까지 삭제   
  ex) git reset --soft HEAD~3 ==> 커밋만 취소하되 최상위에서 3개까지 삭제   
  ex) git reset --mixed e5843d0 ==> 커밋,스테이지 취소하고, 아이디 이후내용 날라감   
  ex) git reset --hard fa815f7 ==> 커밋, 스테이지, 로컬내용 없어지고, 아이디 이후내용 날라감   

git restore aa.txt ==> aa.txt 파일 되돌리기   
git restore --staged hello2.txt  ==> 스테이징 파일을 로컬로 내리기   
git restore --source HEAD aa.txt ==> 현재 커밋(HEAD)에서  aa.txt 가져와서 작업 디렉토리에 복사   
git restore --source 7abc7def aa.txt  ==>  특정파일을 특정커밋시점으로 되돌리기   
   
git revert <커밋 아이디> ==> 커밋한 내용 삭제하되 커밋기록은 복사   
   
- 되돌려야 할 commit이 local 에만 존재할 경우 - reset   
- 되돌려야 할 commit이 push 된 경우 - revert   
- 만약 협업을 하고 있다면, reset 후 push 는 하면 안되는 행동이다. 저장소에 push 하면 에러가 나지만,    
- force 옵션으로 강제로 덮게 된다면 reset 된 커밋으로 덮어지기 때문이다.   





## branch 명령어
git branch ==> 브랜지 목록보기   
git branch test ==> test 브랜치 생성   
git branch  -d test ==> test 브랜지 삭제   
git branch --show-current ==> 현재 사용중인 브랜치만 표시   
   
	git switch test / 브랜치 변경      
	git checkout test / 브랜치 변경   
   
git merge test ==> main에서 test 브랜지 병합, 다른 파일이면 그냥 저장되고,    
                         같은파일 다른내용이면변경내용 확인    
                         같은 파일 같은 내용이면 페이지내에서 색깔로 표시됨   
git cherry-pick 12345 ==> 다른 브랜치의 값을 현재 브랜치에 병합, 커밋아이디 사용함.   
   
   
### stash 임시내용 저장 - 로컬이나 스테이징된것 임시저장   
$ git stash ==> trash 생성      
$ git stash list ==>  trash 목록 보기      
$ git stash pop ==>  넣은내용 꺼내기      
   
   
### github 동기화      
git clone https://github.com/jwp9622/project.git ==> 로컬과 저장소 동기화, 하위 폴더 받음   
git clone https://github.com/jwp9622/project.git . ==> 로컬과 저장소 동기화, 현재폴더에 받음   
git remote add origin https://github.com/jwp9622/test-1.git ==> 로컬과 저장소 동기화, 하위 폴더 받음   
git remote -v ==>  원격저장소 연결확인   
git remote rm origin ==> git origin 삭제   
git push -u orgin main ==> 원격저장소로 main 브랜치올리기,branch 별로 올려야됨.   
git push -u orgin test ==> 원격저장소로 test 브랜치올리기,branch 별로 올려야됨.   
git pull origin main ==> 원격 저장소에서 내려받기, 브랜치별로 받을수 있음.   
git fetch ==> 원격저장소 변경내용 보여줌. 병합 안함.   
   
   
## ssh key 만들기   
$ cd ~ /홈 디렉토리 이동   
$ ssh-keygen -t ed25519 -C "jwp9622@naver.com" / ssh key 만들기   
$ ssh-keygen -t rsa -b 4096 -C "your_email@example.com" / ssh key 만들기   
$ cd .ssh /ssh 로 이동   
$ ls -la / 목록보기   
total 22   
drwxr-xr-x 1 jeongwon 197121   0  5월 12 10:31 ./   
drwxr-xr-x 1 jeongwon 197121   0  5월 12 10:31 ../   
-rw-r--r-- 1 jeongwon 197121 411  5월 12 10:31 id_ed25519 / 프라이빗키 , 로컬에 저장   
-rw-r--r-- 1 jeongwon 197121  99  5월 12 10:31 id_ed25519.pub   / 퍼블릭 키, 원격저장소에 저장   
$ eval "$(ssh-agent -s)" / ssh 등록   
$ ssh-add ~/.ssh/id_ed25519 / 로컬에 등록   
$ clip < ~/.ssh/id_ed25519.pub / 깃허브에 키 복사   
** github dashboard > 사용자이이콘 > SSH and GPG Keys > new SSH key 생성 > 붙여넣기   


## GitHub
GitHub에서 PR(Pull Request)는 main이외의 branch를 push 하면 나타남.   
   
ssh-keygen   
git fetch   
git checkout -b 브랜치명   
git checkout    
   
   
git add .   
git commit -m "파일삭제"   
git push origion main   

git reset --soft HEAD~1   
git reset --mixed HEAD~1   
git reset --hard HEAD~1   
git reset --soft HEAD~1 숫자만큼 커밋   
git reset HEAD^ 최신꺼만  커밋   


