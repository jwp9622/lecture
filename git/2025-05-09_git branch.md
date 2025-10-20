# 2025.5.9 git branch
- 기술은 생계기술이다.   
- 단순한 지식들은 단기간이지만 전문적인 기술은 살아남는다.   
- 훈련의 과정은 살아남는다.   
- 나를 훈련시켜야한다. 어렵고 힘든 상황들에도 좋은 결과가 나올수 있다.   
- 시간많이 걸리고 훈련의 과정이 필요하다.   
- git은 시연이 필요하다   
- 실패에서 얻는게많다. 많이 노력해야한다.   
- 2사 8을 이끈다. 내가 2에 속할수 있어야한다.   
- 도구에 대해서 편집하지마라. 웹, 이클립스, 비주얼코드 여러개 사용   

## git branch 사용이유
- 개발주체가 여러고객사로 제품을 만들수 있기때문에   
--  하나의 가지에서 공정을 만드는것보다 여러프로세서에서 하는것이 좋다.   
예를 여러개의 공장에서 생산하는것과 비슷하다.   

## branch
+ git branch /  branch 확인, 기본은 main 이다   
- git branch apple / apple 브랜드 만든다   
- git log --oneline  / 한줄에 한개씩 보여준다.   
- git switch apple   / 브랜지 변경, 스위치를 한것이다.   
- git add . / 파일을 한꺼번에 스테이지에 옮김   
- git log --oneline --branches - branch   / 한꺼번에 보기   
- git log --oneline --branches --graph     / branch를 그래프로 보기   
- git log main..apple / 메인에는 없고 apple에 있는것만 보여주기   
- git log apple..main /  apple에는 없고 메인에 있는것만 보여주기   

## branch 병합하기
- 뻗어나간다고 좋은게 아니다. 제품은 1개이기 때문에    
- git switch main / 병합하기전에 main으로 swithch 해야함   
- git merge o2 / o2메인 브랜치를 기준으로 가져온다. 안내페이지나오고 닫으면 저장된다. merge도 commit이다   
- 두개의 브랜치에 서로다른문서가 있으면 쉽게 병합가능하지만   
- 두개의 브랜치에서 같은문서를 수정하면 어떻게 병합할것인가   
   
- 같은문서의 다른부분수정 - 병합된 내용이 나옴.   
- 같은문서의 같은부분수정 - 병합후 병합 편집기에서 확인후 수정   
   
- git branch -d o2 / 브랜치 삭제   
- git log --stat / 저장소의 파일목록까지 같이 보여주기   
- git log --oneline --branches --graph   


## git 문제

Ⅰ. 단답형
[문제] 다음 물음에 알맞은 Git 명령어를 간단히 작성하시오.
1. fixed라는 브랜치를 새로 생성하는 명령어는? git branch fixed
2. 커밋 해시와 메시지를 간단히 한 줄로 출력하는 명령어는? git log --oneline
3. 현재 브랜치에서 fixed 브랜치로 전환하는 명령어는? git switch fixed
4. 모든 수정된 파일을 stage에 올리는 명령어는? git add .
5. 커밋 히스토리를 그래프로 출력하는 명령어는? git log --graph
6. 현재 브랜치에 fixed 브랜치를 병합하는 명령어는? git merge fixed
7. 병합이 완료된 fixed 브랜치를 삭제하는 명령어는? git branch -d fixed
8. 커밋 해시 12345를 현재 브랜치에 적용하는 명령어는? git cherry-pick 12345
9. file-10이라는 빈 파일을 생성하는 명령어는? touch file-10
10. 브랜치 목록을 확인하는 명령어는? git branch
11. 현재 사용 중인 브랜치를 확인하는 명령어는? git branch --show-current
12. 수정한 내용을 이전 상태로 되돌리는 명령어는? 
-> git restore . 또는 git checkout -- . 
13. 스테이징 영역에서 파일을 제외하는 명령어는? git reset HEAD 파일명
14. 마지막 커밋 메시지를 수정하는 명령어는? git commit --amend
15. 원격 저장소를 추가하는 명령어는? git remote add origin [URL]
16. 원격 저장소에서 코드를 가져오는 명령어는? git pull origin main
17. 변경 사항을 커밋하는 명령어는?  git commit -m "커밋 메시지"
18. 커밋하지 않은 변경 사항을 취소하는 명령어는? git restore .
 git reset //staged 파일들 unstage
 git checkout . //모든 변경 사항 취소
19. git clean -fdx //추적할 수 없는 모든 파일 제거
20. 커밋 내역을 출력하는 명령어는? git log
21. new.txt 파일을 만들고 커밋까지 완료하는 전체 명령어를 쓰시오.
- touch new.txt
- echo "aaa" >> new.txt / git add new.txt
- git add new.txt
- git commit -m "new"


## Git 명령어 시험 정답 및 해설
Ⅰ. 단답형 정답 및 해설
1. git branch fixed – 브랜치 생성
3. git log --oneline – 간단한 로그 출력
4. git switch fixed – 브랜치 전환
5. git add . – 전체 스테이지에 추가
6. git log --graph – 그래프 로그 출력
7. git merge fixed – 병합
8. git branch -d fixed – 브랜치 삭제
9. git cherry-pick 12345 – 특정 커밋 적용
10. touch file-10 – 빈 파일 생성
11. git branch – 브랜치 목록 확인
12. git branch --show-current – 현재 브랜치
13. git restore . 또는 git checkout -- . – 되돌리기
14. git reset HEAD [파일명] – 스테이지에서 제거
15. git commit --amend – 마지막 커밋 수정
16. git remote add origin [URL] – 원격 저장소 추가
17. git pull origin main – 원격 저장소에서 코드 가져오기
18. git commit -m "메시지" – 커밋
19. git restore . – 변경사항 취소
20. git log – 커밋 내역 출력
21. touch new.txt && git add new.txt && git commit -m "add new.txt" – 전체 작업


## Ⅱ. 객관식
[문제] 다음 보기 중 가장 알맞은 것을 고르시오.   
다음 중 브랜치를 새로 생성하는 명령어는? B   
- A. git init   
- B. git branch fixed   
- C. git merge fixed   
- D. git switch   
   
2. 다음 중 특정 커밋을 현재 브랜치에 적용하는 명령어는?C   
- A. git revert   
- B. git reset   
- C. git cherry-pick   
- D. git switch   
   
3. 병합 완료 후 브랜치를 삭제하는 명령어는?C   
- A. git remove   
- B. git delete   
- C. git branch -d   
- D. git merge --abort   
   
4. git log --oneline 명령어의 설명으로 올바른 것은?B   
- A. 커밋을 되돌리는 명령어   
- B. 커밋을 한 줄로 요약해 출력함   
- C. 원격 저장소에서 데이터를 받아옴   
- D. 파일을 stage에서 제외함   
   
5. 다음 중 커밋 그래프를 확인하는 명령어는?B   
- A. git graph   
- B. git log --graph   
- C. git list   
- D. git switch   
   
6. 수정된 파일을 커밋하는 순서는?C   
- A. git push → git pull → git commit   
- B. git add → git status → git clone   
- C. git add → git commit   
- D. git fetch → git commit   
   
7. 스테이징된 파일을 제거하는 명령어는?B   
- A. git revert   
- B. git reset   
- C. git rm   
- D. git restore   
   
8. 원격 저장소를 추가하는 명령어는?A   
- A. git remote add origin URL   
- B. git push origin   
- C. git fetch   
- D. git remote fetch   
   
9. 새 파일을 생성하는 명령어는?B   
- A. git init file.txt   
- B. touch file.txt   
- C. git add file.txt   
- D. git new file.txt   
   
10. Git 저장소를 초기화하는 명령어는?B   
- A. git clone   
- B. git init   
- C. git install   
- D. git reset   
   
Ⅱ. 객관식 정답 및 해설   
- B – 브랜치 생성   
- C – 특정 커밋 적용   
- C – 병합 후 브랜치 삭제   
- B – 한 줄 요약 로그   
- B – 커밋 그래프   
- C – 올바른 커밋 순서   
- B – 스테이지에서 제거   
- A – 원격 저장소 추가   
- B – 빈 파일 생성   
- B – Git 저장소 초기화   

## Ⅲ. 실습형
[문제] 다음 지시사항에 따라 정확한 Git 명령어를 작성하시오.
1. bugfix 브랜치를 생성하고 전환하시오.
- git branch bugfix   
- git switch bugfix   
2. file-10.txt라는 파일을 생성하시오.
- touch file-10.txt   
3. 해당 파일을 스테이징하고 커밋 메시지를 "add file"로 저장하시오.
- echo "aaa">>file-10.txt   
- git add file-10.txt   
- git commit -m "add file"   
4. fixed 브랜치를 main 브랜치에 병합하시오.
- git switch main   
- git merge bugfix   
5. 병합 완료 후 fixed 브랜치를 삭제하시오.
- git branch -d bugfix   
6. 커밋 해시 12345abcd를 cherry-pick 하시오.
- git cherry-pick 12345abcd   
7. 현재 브랜치의 커밋 로그를 그래프로 출력하시오.
- git log --graph   
8. 브랜치 목록을 확인하시오.
- git branch   
9. main 브랜치로 전환하시오.
- git switch main   
10. 현재 디렉토리의 모든 변경 파일을 stage에 추가하시오.
- git add .   

Ⅲ. 실습형 정답 및 해설   
- git branch bugfix && git switch bugfix   
- touch file-10.txt   
- git add file-10.txt && git commit -m "add file"   
- git switch main && git merge fixed   
- git branch -d fixed   
- git cherry-pick 12345abcd   
- git log --graph   
- git branch   
- git switch main   
- git add .   
