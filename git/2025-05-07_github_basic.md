# 2025.5.7 git 기본 명령어

## 솔루션 모델
3-Tier 아키텍처
클라이언트      프리젠테이션          비즈니스      안티그레이션     데이터리소스
----------------------------------------------------------------------------------------
요청(URL)        컨트롤러(servlet)   서비스(java)      라이브러리       데이터베이스
응답(HTML/      뷰(jsp)               엔티티(java)
CSS/JS)



## 소프트웨어 개발 패러다임
구조적           정보공학               객체지향
---------------------------------------------------------
기능               데이터                기능+데이터
                                             클래스(구조체) --> *캡슐화(데이터감추고, 기능위주)
                                             기능 --> 멤버메소드
                                             데이터 --> 멤버필드
  -                                           자료은닉 --> 데이터 무결성 지켜줌.
객체는 처음부터 만들어지는것 아님. 클래스로 만들어짐

## 클래스 vs 객체
1. 저장위치
2. 값
클래스는 소스파일(하드디스크), 객체-전자신호(메모리의 위치)
클래스는 설계도, 객체는 실제값을 가짐

git, github 개발의 모든 산출물을 관리하기 위한 도구

## 형상관리 --> 대표적인 도구 git
형상 --> 개발과정에서 발생된 모든 산출물(관리, 수정, 삭제 전반적인 작업)
관리 --> CRUD(Create Read Update Delete), (생성, 읽기, 변경, 삭졔

## 윈도우에 깃 설치하기
https://desktop.github.com/download/

1. 다운로드
https://git-scm.com/downloads/win  -> download 후 설치 / 에디터에서 visual code 선택
visual code 프로그램 설치

2. 실행
  Git Bash 실행
  코맨드 창에서 git 엔터 -> 설명문구 나오면 설치 완료

## 맥에 깃 설치하기
  1. 다운로드
     https://brew.sh

  2. 홈브류 설치하기
     /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

  3. 홈브류에서 깃 설치하기
     brew install git
     
## git 환경설정
Git Bash 실행
커멘드창에서 아래 명령어 실행
git config --global user.name "dahee"
git config --global user.email "jwp9622@naver.com"

## 리눅스 명령어
- ~ 현재 디렉토리
- pwd 현재 위치 경로
- ls  현재 디렉토리 목록
- ls -l 자세한 디렉토리 리스트
- ls -a 숨겨진 디렉토리
- ls -r 파일의 정렬순서 거꾸로 보여주기
- ls -t 파일 작성시간을 내림차순으로 
- clear 화면 지우기
- cd 디렉토리 이동
- cd . 현재디렉토리
- cd .. 상위디렉토리
- cd ~ 홈디렉토리
- . 상위디렉토리
- .. 상위 디렉토리
- mkdir 디렉토리 생성
- rm -r test -> test 디렉토리 지우기
- exit 터미털 종료

## 명령어 테스트
1. git config user.name "easys" : 깃 환경에서 이름올 'easys'로 지정
2. gif config user.email "doit@naver.com" :  깃 환경에서 이메일을 'doit@naver.com'로 지정한다.
3. pwd : 현재 경로로 표시한다.
4. ls : 현재 디렉토리 안의 내용을 표시한다.
5. ls -a : 현재 디렉토리안의 파일와 폴더 상세 정보까지 표시한다.
6. cd .. : 부모 디렉토리로 이동한다.
7. ls -l : 현재 디렉토리 안의 숨긴 파일과 숨긴 디렉토리도 표시한다.
8. clear : 화면을 깨끗하게 지운다.
9. cd 경로 :하위 디렉토리로 이동한다.
10. mkdir : 새 디렉토리를 만든다.
11. cd ~ : 홈 디렉토리로 이동한다.
12. rm -r 파일명: 파일이나 디렉토리를 삭제한다.
13. exit : 터미널 창을 종료한다.
