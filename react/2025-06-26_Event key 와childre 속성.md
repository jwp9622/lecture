# Event key와 childre 속성

모든 리액트 컴포넌트 --> key, ref, children 구성되어 있다.   
   
* key 속성 설정하기
  리액트 프레임워크 -->` <p>`   
  App.tsx   


* Event

| 속성 이름           | 설명                                                                   |
| --------------- | -------------------------------------------------------------------- |
| `type`          | 이벤트의 이름 (예: `"click"`, `"keydown"` 등)                                |
| `isTrusted`     | 이벤트가 브라우저에 의해 발생했는지 여부<br>→ `true`: 브라우저가 생성<br>→ `false`: JS 코드로 생성 |
| `target`        | 이벤트가 실제로 발생한 DOM 요소                                                  |
| `currentTarget` | 이벤트 리스너가 등록된 DOM 요소<br>→ 이벤트 버블링/캡처 중 현재 위치한 요소                      |
| `bubbles`       | 이벤트가 버블링 되는 이벤트인지 여부 (`true`/`false`)                                |



* EventTarget 타입

  EventTarget   
  Node  
  Element  
  HTMLElement  



* 이벤트 처리기   
  
  DOM_객체.addEventListener(이벤트_이름 : string, 콜백_함수 : (e:Event)  => void)   


* 델리게이션 모델   
        Event              Event       
Event Source  ───▶  Listener  ───▶  Handler   
(발생)&nbsp;&nbsp;&nbsp;(감지)           (처리)   



* React 이벤트 컴포넌트 요약

| 컴포넌트 이름                   | 설명 (다루는 이벤트 개념)                                                                                                     |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **`<FileDrop />`**        | 🔽 **Drag & Drop**으로 파일을 드롭하는 영역을 구현<br>→ `onDragOver`, `onDrop` 이벤트 사용<br>→ 드래그한 파일을 직접 브라우저에 올릴 수 있음              |
| **`<DragDrop />`**        | 🔁 HTML5 Drag and Drop 기본 예제<br>→ `onDragStart`, `onDrop`, `onDragOver`, `onDragLeave` 등 사용<br>→ 요소를 끌어서 다른 위치로 옮기기 |
| **`<FileInput />`**       | 📁 파일을 `<input type="file" />`을 통해 업로드<br>→ `onChange` 이벤트 사용<br>→ 사용자가 파일을 선택했을 때 처리                               |
| **`<OnChange />`**        | ✏️ 입력값을 입력할 때마다 상태 변경<br>→ `onChange` 이벤트 사용<br>→ `input`, `textarea`, `select` 등에서 값 추적                            |
| **`<VariousInputs />`**   | ✅ 여러 입력 필드들의 onChange 처리<br>→ 텍스트, 체크박스, 라디오 버튼 등 다양한 input을 제어<br>→ `onChange` + 조건 분기                             |
| **`<StopPropagation />`** | ⛔ 이벤트 전파 중단<br>→ `event.stopPropagation()` 사용<br>→ 부모 컴포넌트로 이벤트가 퍼지는 것 방지                                           |
| **`<EventBubbling />`**   | 🧼 이벤트 버블링 개념 설명<br>→ 자식 요소에서 발생한 이벤트가 부모로 올라가는 것<br>→ 디버깅/UX 처리 시 중요한 개념                                           |
| **`<DispatchEvent />`**   | 🚀 JS에서 이벤트를 수동으로 발생시키기<br>→ `element.dispatchEvent()`로 인위적인 이벤트 트리거<br>→ 테스트나 특별한 로직에서 사용                          |
| **`<ReactOnClick />`**    | 🖱️ `onClick`을 통한 클릭 이벤트 처리<br>→ React의 Synthetic Event 시스템 기반<br>→ 가장 기본적인 사용자 인터랙션 처리                             |
| **`<OnClick />`**         | 🖱️ 일반 클릭 이벤트 예제<br>→ React의 `onClick` 또는 직접 DOM 이벤트 비교 가능성 있음                                                      |
| **`<EventListener />`**   | 🎧 직접 `addEventListener`로 DOM 이벤트 등록<br>→ React 외부의 전통적인 JS 방식<br>→ `useEffect`에서 DOM 이벤트 수동 등록/제거                  |




## 실습 예제 : map 연습

>npx create-react-app ch02_3 --template typescript   

>code ch02_3

터미널 > 새터미널

>xcopy  ..\..\ch02\ch02_3\src\* .\src\ /E /I 

>xcopy  ..\..\ch02\ch02_3\* .\  

>npm i chance luxon

>npm i -D @types/chance @types/luxon 

>npm start

* src/App.tsx

		import {Component} from 'react'
		import React, { useState } from 'react';
		import ClassComponent from './ClassComponent'
		import ArrowComponent from './ArrowComponent'
		import ProfileCardComponent from './ProfileCardComponent'
		import P from './P'

		export default function App() {
		  const texts = ['hello', 'world'].map((text, index) => 
		  <P key={index} children={text} />
		)
		  return <div children={texts} />
		}


* src/P.tsx

		import type {FC, PropsWithChildren } from 'react'

		export type PProps = {}
		const P: FC<PropsWithChildren<PProps>> = props =>{
		  return <p {...props} />
		}
		export default P



## 실습예제2 : Event 연습

>npx create-react-app ch02_3 --template typescript   

>code ch02_3

터미널 > 새터미널

>xcopy  ..\..\ch02\ch02_3\src\* .\src\ /E /I 

>xcopy  ..\..\ch02\ch02_3\* .\  

>npm i chance luxon

>npm i -D @types/chance @types/luxon 

>npm start


>touch src/copy/copyMe.tsx

>xcopy copy\CopyMe.tsx pages\EventListener.tsx /E /I

>xcopy copy\CopyMe.tsx pages\OnClick.tsx

>xcopy copy\CopyMe.tsx pages\ReactOnClick.tsx

>xcopy copy\CopyMe.tsx pages\EventBubbling.tsx

>xcopy copy\CopyMe.tsx pages\StopPropagation.tsx

>xcopy copy\CopyMe.tsx pages\DispatchEvent.tsx

>xcopy copy\CopyMe.tsx pages\VariousInputs.tsx

>xcopy copy\CopyMe.tsx pages\OnChange.tsx

>xcopy copy\CopyMe.tsx pages\FileInput.tsx

>xcopy copy\CopyMe.tsx pages\DragDrop.tsx

>xcopy copy\CopyMe.tsx pages\FileDrop.tsx


* src/pages/App.tsx

		import EventListener from './pages/EventListener'
		import OnClick from './pages/OnClick'
		import DispatchEvent from './pages/DispatchEvent'
		import EventBubbling from './pages/EventBubbling'
		import StopPropagation from './pages/StopPropagation'
		import VariousInputs from './pages/VariousInputs'
		import OnChange from './pages/OnChange'
		import FileInput from './pages/FileInput'
		import DragDrop from './pages/DragDrop'
		import FileDrop from './pages/FileDrop'
		import ReactOnClick from './pages/React_OnClick'

		export default function App() {
		  return (
		    <div>gggg
		      <FileDrop />
		      <DragDrop />
		      <FileInput />
		      <OnChange />
		      <VariousInputs />
		      <StopPropagation />
		      <EventBubbling />
		      <DispatchEvent />
		      <ReactOnClick />
		      <OnClick />
		      <EventListener />
		    </div>
		  )
		}

* src/pages/DispatchEvent.tsx

		import React, { useRef, useEffect } from 'react';

		export default function DispatchEvent() {

		  // 👇 버튼 DOM 요소 타입 명시
		  const realButtonRef = useRef<HTMLButtonElement>(null);

		  // 클릭 핸들러 함수 정의 위치 위로 옮기기
		  const handleRealClick = () => {
		    alert('🎯 실제 버튼이 클릭되었습니다!');
		  };

		  const handleFakeClick = () => {
		    if (realButtonRef.current) {
		      const event = new MouseEvent('click', {
			bubbles: true,
			cancelable: true,
		      });
		      realButtonRef.current.dispatchEvent(event); // ✅ 타입 오류 없음
		    }
		  };

		  useEffect(() => {
		    const btn = realButtonRef.current;
		    if (btn) {
		      btn.addEventListener('click', handleRealClick);
		    }
		    return () => {
		      if (btn) {
			btn.removeEventListener('click', handleRealClick);
		      }
		    };
		  }, []);


		  return (
		    <div style={{ padding: '2rem' }}>
		      <h2>🔵 dispatchEvent 예제 (React)</h2>
		      <button ref={realButtonRef} style={{ marginRight: '1rem' }}>
			실제 버튼
		      </button>
		      <button onClick={handleFakeClick}>이벤트 강제 발생 (dispatchEvent)</button>
		    </div>
		  );


		  //return <div>CopyMe</div>
		}


* src/pages/DragDrop.tsx

		import React, { useState } from 'react';
		export default function CopyMe() {

		  const [dragOver, setDragOver] = useState(false);
		  //const [droppedItem, setDroppedItem] = useState(null);
		  const [droppedItem, setDroppedItem] = useState<string | null>(null);

		  const handleDragStart = (e: React.DragEvent<HTMLDivElement>) => {
		    e.dataTransfer.setData('text/plain', '🍎 사과');
		  };

		  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
		    e.preventDefault(); // 필수! drop 허용
		    setDragOver(true);
		  };

		  const handleDragLeave = () => {
		    setDragOver(false);
		  };

		  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
		    e.preventDefault();
		    const data = e.dataTransfer.getData('text/plain');
		    setDroppedItem(data);
		    setDragOver(false);
		  };

		  return (
		    <div style={{ padding: '2rem' }}>
		      <h2>🟢 React Drag & Drop 예제</h2>

		      {/* 드래그 가능한 요소 */}
		      <div
			draggable
			onDragStart={handleDragStart}
			style={{
			  width: '100px',
			  padding: '1rem',
			  backgroundColor: '#f9c74f',
			  cursor: 'grab',
			  borderRadius: '8px',
			  marginBottom: '1rem',
			  textAlign: 'center',
			}}
		      >
			🍎 드래그할 아이템
		      </div>

		      {/* 드롭 대상 */}
		      <div
			onDragOver={handleDragOver}
			onDragLeave={handleDragLeave}
			onDrop={handleDrop}
			style={{
			  width: '200px',
			  height: '120px',
			  border: '2px dashed #aaa',
			  backgroundColor: dragOver ? '#90be6d' : '#f0f0f0',
			  display: 'flex',
			  alignItems: 'center',
			  justifyContent: 'center',
			  fontSize: '1.2rem',
			  borderRadius: '10px',
			  transition: 'background-color 0.2s',
			}}
		      >
			{droppedItem ? `드롭됨: ${droppedItem}` : '여기로 드래그'}
		      </div>
		    </div>
		  );



		  //return <div>CopyMe</div>
		}


* src/pages/EventBubbling.tsx

		import type {SyntheticEvent} from 'react'

		export default function EventBubbling() {
		  const onDivClick = (e: SyntheticEvent) => {
		    const {isTrusted, target, bubbles, currentTarget} = e
		    console.log('click event bubbles on <div>', isTrusted, target, bubbles, currentTarget)
		  }
		  const onButtonClick = (e: SyntheticEvent) => {
		    const {isTrusted, target, bubbles} = e
		    console.log('click event starts at <button>', isTrusted, target, bubbles)
		  }
		  return (
		    <div onClick={onDivClick}>
		      <p>EventBubbling</p>
		      <button onClick={onButtonClick}>Click Me</button>
		    </div>
		  )
		}



* src/pages/EventListener.tsx

		document.getElementById('root')?.addEventListener('click', (e: Event) => {
		  const {isTrusted, target, bubbles} = e
		  console.log('mouse click occurs.', isTrusted, target, bubbles)
		})
		document.getElementById('root')?.addEventListener('click', (e: Event) => {
		  const {isTrusted, target, bubbles} = e
		  console.log('mouse click also occurs.', isTrusted, target, bubbles)
		})

		export default function EventListener() {
		  return <div>EventListener</div>
		}



* src/pages/FileDrop.tsx

		import type {DragEvent} from 'react'

		export default function FileDrop() {
		  const onDragOver = (e: DragEvent) => e.preventDefault()

		  const onDrop = (e: DragEvent) => {
		    e.preventDefault() // 웹 브라우저의 새로운 창에 드롭한 이미지가 나타나는 것을 방지
		    const files = e.dataTransfer.files
		    if (files) {
		      for (let i = 0; i < files.length; i++) {
			const file: File | null = files.item(i) //혹은 file = files[i];
			console.log(`file[${i}]: `, file)
		      }
		    }
		  }

		  return (
		    <div>
		      <p>FileDrop</p>
		      <div onDrop={onDrop} onDragOver={onDragOver}>
			<h1>Drop image files over Me</h1>
		      </div>
		    </div>
		  )
		}



* src/pages/FileInput.tsx

		import type {ChangeEvent} from 'react'

		export default function FileInput() {
		  const onChange = (e: ChangeEvent<HTMLInputElement>) => {
		    const files: FileList | null = e.target.files
		    if (files) {
		      for (let i = 0; i < files.length; i++) {
			const file: File | null = files.item(i) //or    file = files[i];
			console.log(`file[${i}]: `, file)
		      }
		    }
		  }
		  return (
		    <div>
		      <p>FileInput</p>
		      <input type="file" onChange={onChange} multiple accept="image/*" />
		    </div>
		  )
		}


* src/pages/OnChange.tsx

		import React, { useState } from 'react';

		export default function OnChange() {
		  const [value, setValue] = useState('');

		  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
		    setValue(e.target.value);
		  };

		  return (
		    <div style={{ padding: '1rem' }}>
		      <h3>✏️ 입력값: {value}</h3>
		      <input type="text" value={value} onChange={handleChange} />
		    </div>
		  );


		  //return <div>CopyMe</div>
		}



* src/pages/OnClick.tsx

		const rootDiv = document.getElementById('root')
		if (rootDiv) {
		  rootDiv.onclick = (e: Event) => {
		    const {isTrusted, target, bubbles} = e
		    console.log('mouse click occurs on rootDiv', isTrusted, target, bubbles)
		  }
		  rootDiv.onclick = (e: Event) => {
		    const {isTrusted, target, bubbles} = e
		    // prettier-ignore
		    console.log('mouse click also occurs on rootDiv', isTrusted, target, bubbles)
		  }
		}
		export default function OnClick() {
		  return <div>OnClick</div>
		}


* src/pages/React_OnClick.tsx

		import type {SyntheticEvent} from 'react'
		export default function ReactOnClick() {
		  const onClick = (e: SyntheticEvent) => {
		    const {isTrusted, target, bubbles} = e
		    console.log('mouse click occurs on <button>', isTrusted, target, bubbles)
		  }
		  return (
		    <div>
		      <p>ReactOnClick</p>
		      <button onClick={onClick}>Click Me</button>
		    </div>
		  )
		}

* src/pages/StopPropagation.tsx

		import type {SyntheticEvent} from 'react'

		export default function StopPropagation() {
		  const onDivClick = (e: SyntheticEvent) => console.log('click event bubbles on <div>')
		  const onButtonClick = (e: SyntheticEvent) => {
		    console.log('mouse click occurs on <button> and call stopPropagation')
		    e.stopPropagation()
		  }
		  return (
		    <div onClick={onDivClick}>
		      <p>StopPropagation</p>
		      <button onClick={onButtonClick}>Click Me and stop event propagation</button>
		    </div>
		  )
		}



* src/pages/VariousInputs.tsx

		export default function VariousInputs() {
		  return (
		    <div>
		      <p>VariousInputs</p>
		      <div>
			<input type="text" placeholder="enter some texts" />
			<input type="password" placeholder="enter your password" />
			<input type="email" placeholder="enter email address" />
			<input type="range" />
			<input type="button" value="I'm a button" />
			<input type="checkbox" value="I'm a checkbox" defaultChecked />
			<input type="radio" value="I'm a radio" defaultChecked />
			<input type="file" />
		      </div>
		    </div>
		  )
		}
