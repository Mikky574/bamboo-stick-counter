/* Behavioral unit checks of the shipped page script; no browser automation. */
const fs=require('node:fs');
const vm=require('node:vm');
const assert=require('node:assert/strict');
const html=fs.readFileSync('templates/index.html','utf8');
const script=html.match(/<script>([\s\S]*?)<\/script>/)[1];

function page(){
 const elements=new Map(),events=new Map(),calls=[],timers=new Map();let sequence=0,activeElement=null;
 function listen(target,name,fn){const previous=target.get(name);target.set(name,previous?(event=>{previous(event);fn(event)}):fn)}
 for(const match of html.matchAll(/<([a-z][\w:-]*)\b([^>]*\bid="([^"]+)"[^>]*)>/gi)){
  const id=match[3],source=match[2];assert(!elements.has(id),'Duplicate DOM id '+id);
  const listeners=new Map(),attributes=new Map(Array.from(source.matchAll(/([\w-]+)="([^"]*)"/g),m=>[m[1],m[2]])),classes=new Set(),captures=new Set();
  const element={id,tagName:match[1].toUpperCase(),nodeName:match[1].toUpperCase(),type:attributes.get('type')||'',hidden:/\bhidden(?:\s|$)/.test(source),disabled:/\bdisabled(?:\s|$)/.test(source),checked:/\bchecked(?:\s|$)/.test(source),textContent:'',value:attributes.get('value')||'',min:attributes.get('min')||'',max:attributes.get('max')||'',style:{setProperty(name,value){this[name]=value}},dataset:{},offsetWidth:180,
   classList:{add(...names){names.forEach(name=>classes.add(name))},remove(...names){names.forEach(name=>classes.delete(name))},toggle(name,force){const add=force===undefined?!classes.has(name):force;if(add)classes.add(name);else classes.delete(name);return add},contains(name){return classes.has(name)}},addEventListener(name,fn){listen(listeners,name,fn)},listeners,
   setAttribute(name,value){attributes.set(name,String(value))},getAttribute(name){return attributes.get(name)??null},removeAttribute(name){attributes.delete(name)},
   focus(){activeElement=this},blur(){if(activeElement===this)activeElement=null},setPointerCapture(id){captures.add(id)},releasePointerCapture(id){captures.delete(id)},hasPointerCapture(id){return captures.has(id)},
   getBoundingClientRect(){return{left:0,top:0,right:800,bottom:500,width:800,height:500}},width:Number(attributes.get('width'))||360,height:Number(attributes.get('height'))||360};
  const context={canvas:element,commands:[],measureText(text){return{width:String(text).length*8}}};
  for(const name of ['drawImage','clearRect','beginPath','closePath','moveTo','lineTo','stroke','arc','ellipse','rect','roundRect','fill','fillRect','strokeRect','strokeText','fillText','save','restore','clip','translate','scale','setTransform','setLineDash'])context[name]=(...args)=>context.commands.push([name,...args]);
  element.context=context;element.getContext=()=>context;elements.set(id,element);
 }
 const state={response:null,width:800,height:500,readers:[],fetchCalls:0,now:0};
 function schedule(fn,delay,interval){const id=++sequence;timers.set(id,{fn,delay,at:state.now+delay,interval});return id}
 async function flush(){for(let i=0;i<16;i++)await Promise.resolve()}
 async function advance(ms){const end=state.now+ms;while(true){const next=Array.from(timers.entries()).filter(([,task])=>task.at<=end).sort((a,b)=>a[1].at-b[1].at)[0];if(!next)break;const [id,task]=next;state.now=task.at;if(task.interval)task.at+=task.delay;else timers.delete(id);task.fn();await flush()}state.now=end;await flush()}
 const browser={console,performance:{now:()=>state.now},Number,Math,Array,JSON,Promise,Error,
  document:{getElementById(id){assert(elements.has(id),'Unknown DOM id '+id);return elements.get(id)},get activeElement(){return activeElement},addEventListener(name,fn){listen(events,name,fn)}},
  setInterval(fn,delay){return schedule(fn,delay,true)},clearInterval(id){timers.delete(id)},
  requestAnimationFrame(fn){queueMicrotask(fn);return ++sequence},cancelAnimationFrame(){},
  setTimeout(fn,delay){return schedule(fn,delay,false)},clearTimeout(id){timers.delete(id)},
  AbortController:class{constructor(){const listeners=new Set();this.signal={aborted:false,addEventListener(name,fn){if(name==='abort')listeners.add(fn)},removeEventListener(name,fn){if(name==='abort')listeners.delete(fn)}};this.listeners=listeners}abort(){if(this.signal.aborted)return;this.signal.aborted=true;for(const fn of this.listeners)fn()}},
  fetch:async(url,options)=>{state.fetchCalls++;calls.push({url,options});return await state.response(url,options)},
  FileReader:class{readAsDataURL(file){this.result=file.url||'data:image/png;base64,AA';state.readers.push(this);queueMicrotask(()=>this.onload())}},
  Image:class{set src(value){this.naturalWidth=state.width;this.naturalHeight=state.height;queueMicrotask(()=>this.onload())}},
 };
 browser.window=browser;browser.addEventListener=(name,fn)=>listen(events,name,fn);
 vm.createContext(browser);vm.runInContext(script,browser);
 function evaluate(code){return vm.runInContext(code,browser)}
 function fixture(result,status=200){state.response=async()=>({ok:status>=200&&status<300,status,json:async()=>JSON.parse(JSON.stringify(result))})}
 async function load(){await evaluate('selectFile({name:"photo.png",type:"image/png",size:2048})')}
 async function run(result,status=200){fixture(result,status);await elements.get('run').onclick()}
 return{elements,events,calls,state,browser,timers,evaluate,fixture,load,run,advance,flush,get:id=>elements.get(id)};
}
function result(count=2,meta={}){return{count,width:800,height:500,points:Array.from({length:count??0},(_,i)=>({x:10+i*3,y:20})),meta:{prediction_valid:true,...meta}}}
function enableExemplars(p){p.get('exemplar-enabled').checked=true;p.get('exemplar-enabled').onchange()}
function pointer(x,y,options={}){return{clientX:x,clientY:y,button:0,pointerId:1,pointerType:'mouse',preventDefault(){},...options}}
function fire(p,id,name,event={}){const element=p.get(id),input={target:element,currentTarget:element,preventDefault(){},...event};element['on'+name]?.(input);element.listeners.get(name)?.(input)}
function tap(p,x,y,options={}){const event=pointer(x,y,options);fire(p,'canvas','pointerdown',event);fire(p,'canvas','pointerup',event);p.events.get('pointerup')?.(event)}
function drag(p,from,to,options={}){const first=pointer(...from,options),last=pointer(...to,options);fire(p,'canvas','pointerdown',first);fire(p,'canvas','pointermove',last);fire(p,'canvas','pointerup',last);p.events.get('pointerup')?.(last)}
function key(p,value,options={}){let prevented=false;const event={key:value,target:p.get('canvas'),ctrlKey:false,metaKey:false,shiftKey:false,altKey:false,preventDefault(){prevented=true},...options};p.events.get('keydown')?.(event);return prevented}
function range(p,value,commit=true){p.get('exemplar-size').value=String(value);fire(p,'exemplar-size','input');if(commit)fire(p,'exemplar-size','change')}
async function test(name,fn){await fn();console.log('PASS '+name)}
(async()=>{
 await test('upload readiness and image-only request',async()=>{
  const p=page();await p.load();assert.equal(p.get('run').disabled,false);
  await p.run(result());assert.equal(p.get('count').textContent,'2');assert.equal(p.calls.length,1);
  assert.equal(p.calls[0].url,'/api/count');assert.deepEqual(Object.keys(JSON.parse(p.calls[0].options.body)),['image']);
  assert.equal(p.get('result-card').dataset.state,'success');assert.equal(p.get('spinner').hidden,true);assert.equal(p.timers.size,0);
 });
 await test('GPU detector results remain candidates, not verified totals',async()=>{
  const p=page();await p.load();await p.run(result(2,{framework:'CountGD++',category:'cotton',device:'cuda',model_load_seconds:6}));
  assert.equal(p.get('result-card').dataset.state,'candidate');assert.equal(p.get('count-label').textContent,'检测候选数');
  assert.match(p.get('image-badge').textContent,/GPU/);
 });
 await test('sharp candidate wording and accepted scene fallback',async()=>{
  const p=page();await p.load();await p.run(result(2,{count_kind:'sharp_candidates',scene_classification:{accepted:true,scene:'sharp'}}));
  assert.equal(p.get('count-label').textContent,'可见尖头候选');assert.equal(p.get('category').textContent,'竹签尖头');
 });
 await test('brown rice fallback preserves candidate scope and review message',async()=>{
  const p=page();await p.load();await p.run(result(3,{framework:'CountGD++',category:'rice',count_kind:'grain_candidates',count_is_total:false,review_required:true,user_message:'含断粒和图边截断部分，请核对'}));
  assert.equal(p.get('count-label').textContent,'可见谷粒候选');assert.equal(p.get('count-unit').textContent,'个');
  assert.equal(p.get('result-card').dataset.state,'candidate');assert.equal(p.get('warning').textContent,'含断粒和图边截断部分，请核对');
 });
 await test('visible chopstick candidates retain their limited scope and specific message',async()=>{
  const p=page();await p.load();await p.run(result(4,{count_kind:'visible_rod_candidates',category:'laid_rods',count_is_total:false,user_message:'上端未入镜，只数可见端面'}));
  assert.equal(p.get('count-label').textContent,'可见筷子候选');assert.equal(p.get('count-unit').textContent,'根');
  assert.equal(p.get('result-card').dataset.state,'candidate');assert.equal(p.get('warning').textContent,'上端未入镜，只数可见端面');
  assert.match(p.get('caption').textContent,/未完整入镜/);
 });
 await test('uncertain result clears old markers and never shows zero',async()=>{
  const p=page();await p.load();await p.run(result());await p.run(result(null,{prediction_valid:false,user_message:'贴合处无法分清',scene_classification:{accepted:true,scene:'laid_rods'}}));
  assert.equal(p.get('count').textContent,'—');assert.equal(p.evaluate('points.length'),0);assert.equal(p.get('result-card').dataset.state,'uncertain');assert.equal(p.get('warning').textContent,'贴合处无法分清');assert.equal(p.get('category').textContent,'平铺筷子 / 木杆');
 });
 await test('zero failure and service error do not retain a previous count',async()=>{
  const p=page();await p.load();await p.run(result());await p.run(result(0,{diagnosis:'no_initial_exemplars'}));assert.equal(p.get('count').textContent,'—');
  await p.run({error:'模型暂不可用'},503);assert.equal(p.get('count').textContent,'—');assert.equal(p.get('result-card').dataset.state,'error');assert.equal(p.get('warning').textContent,'模型暂不可用');assert.equal(p.get('run').disabled,false);assert.equal(p.timers.size,0);
 });
 await test('large photos and invalid uploads fail before inference',async()=>{
  const p=page();p.state.width=8000;p.state.height=5000;await p.load();assert.equal(p.evaluate('dataURL'),'');assert.equal(p.get('canvas').hidden,true);assert.equal(p.get('run').disabled,true);assert.match(p.get('warning').textContent,/3000/);assert.equal(p.state.fetchCalls,0);
  await p.evaluate('selectFile({name:"too-large.jpg",type:"image/jpeg",size:21*1024*1024})');assert.match(p.get('warning').textContent,/20 MB/);
 });
 await test('dense main view hides numbers; rod axes and original hold remain functional',async()=>{
  const p=page();await p.load();await p.run(result(121));let commands=p.get('canvas').context.commands;
  assert.equal(commands.filter(c=>c[0]==='fillText').length,0);assert.equal(p.evaluate('points.length'),121);
  fire(p,'canvas','pointerdown',pointer(10,10));assert.equal(p.evaluate('originalHeld'),true);p.events.get('pointerup')(pointer(10,10));assert.equal(p.evaluate('originalHeld'),false);
  const r=result(1);r.points[0].segment=[[5,5],[90,100]];await p.run(r);commands=p.get('canvas').context.commands;assert(commands.some(c=>c[0]==='lineTo'));
 });
 await test('stale asynchronous replies cannot overwrite a newer revision',async()=>{
  const p=page();await p.load();let deliver;p.state.response=()=>new Promise(resolve=>deliver=resolve);
  const running=p.get('run').onclick();p.evaluate('revision++; clearResult(); setBusy(false)');deliver({ok:true,status:200,json:async()=>result(9)});await running;
  assert.equal(p.get('count').textContent,'—');assert.equal(p.evaluate('points.length'),0);assert.equal(p.timers.size,0);
 });
 await test('resource shortage preserves the image and does not blame photo clarity',async()=>{
  const p=page();await p.load();await p.run(result(2));await p.run(result(null,{prediction_valid:false,diagnosis:'gpu_resources_unavailable',retryable:true,user_message:'请等待其他 GPU 任务释放资源后再试。'}));
  assert.equal(p.get('count').textContent,'—');assert.equal(p.evaluate('points.length'),0);assert.equal(p.get('result-state').textContent,'资源暂不可用');
  assert.match(p.get('status').textContent,/GPU 资源暂不可用/);assert.doesNotMatch(p.get('image-badge').textContent,/清晰/);assert.equal(p.get('run').disabled,false);assert.notEqual(p.evaluate('dataURL'),'');
 });
 await test('100 second timeout aborts waiting, clears old results and never retries automatically',async()=>{
  const p=page();await p.load();await p.run(result(2));let deliver;p.state.response=()=>new Promise(resolve=>deliver=resolve);
  const running=p.get('run').onclick();await p.flush();const pendingCall=p.calls.at(-1);
  assert.equal(p.get('run').disabled,true);await p.get('run').onclick();assert.equal(p.state.fetchCalls,2);
  await p.advance(10000);assert.match(p.get('status').textContent,/仍在处理图片/);
  await p.advance(10000);assert.match(p.get('status').textContent,/仍在等待服务返回/);
  await p.advance(80000);await running;
  assert.equal(pendingCall.options.signal.aborted,true);assert.equal(p.get('result-state').textContent,'等待超时');assert.match(p.get('warning').textContent,/后台任务可能仍在处理/);
  assert.equal(p.get('count').textContent,'—');assert.equal(p.evaluate('points.length'),0);assert.equal(p.get('run').disabled,false);assert.equal(p.get('file').disabled,false);assert.equal(p.get('spinner').hidden,true);assert.equal(p.timers.size,0);assert.equal(p.state.fetchCalls,2);
  deliver({ok:true,status:200,json:async()=>result(99)});await p.flush();assert.equal(p.get('count').textContent,'—');
  await p.run(result(3));assert.equal(p.get('count').textContent,'3');assert.equal(p.state.fetchCalls,3);
 });
 await test('timeout covers a response whose JSON body never finishes',async()=>{
  const p=page();await p.load();let deliverJSON;p.state.response=async()=>({ok:true,status:200,json:()=>new Promise(resolve=>deliverJSON=resolve)});
  const running=p.get('run').onclick();await p.flush();assert.equal(typeof deliverJSON,'function');await p.advance(100000);await running;
  assert.equal(p.get('result-state').textContent,'等待超时');assert.equal(p.get('run').disabled,false);assert.equal(p.timers.size,0);
  await p.run(result(3));deliverJSON(result(99));await p.flush();assert.equal(p.get('count').textContent,'3');
 });
 await test('an older deadline cannot clear or unlock a newer request',async()=>{
  const p=page();await p.load();p.state.response=()=>new Promise(()=>{});const older=p.get('run').onclick();await p.flush();await p.advance(1000);
  p.evaluate('revision++; clearResult(); setBusy(false)');let deliver;p.state.response=()=>new Promise(resolve=>deliver=resolve);const newer=p.get('run').onclick();await p.flush();
  await p.advance(99000);await older;assert.equal(p.get('run').disabled,true);assert.equal(p.get('result-card').dataset.state,'busy');assert.equal(p.calls.at(-1).options.signal.aborted,false);
  deliver({ok:true,status:200,json:async()=>result(3)});await newer;assert.equal(p.get('count').textContent,'3');assert.equal(p.get('run').disabled,false);assert.equal(p.timers.size,0);
 });
 await test('optional reference selection maps native pixels and ignores canvas margins',async()=>{
  const p=page();p.state.width=3000;p.state.height=2000;await p.load();
  assert.equal(p.get('exemplar-enabled').checked,false);assert.equal(p.get('run-exemplars').disabled,true);
  assert.equal(p.evaluate('exemplarDiameter'),100);enableExemplars(p);
  p.get('canvas').getBoundingClientRect=()=>({left:170,top:90,width:600,height:400});
  tap(p,470,290);assert.equal(p.evaluate('exemplarPoints[0].x'),1500);assert.equal(p.evaluate('exemplarPoints[0].y'),1000);
  tap(p,169,290);tap(p,770,290);tap(p,470,89);tap(p,470,490);
  assert.equal(p.evaluate('exemplarPoints.length'),1);assert.equal(p.get('run-exemplars').disabled,false);
  range(p,60);assert.equal(p.evaluate('exemplarDiameter'),60);assert.equal(p.evaluate('exemplarPoints[0].diameter'),60);
  assert(p.get('canvas').context.commands.some(box=>box[0]==='strokeRect'&&Math.abs(box[3]-40)<0.01&&Math.abs(box[4]-40)<0.1));
  assert(p.get('canvas').context.commands.some(c=>c[0]==='fillText'&&c[1]==='E1'));
 });
 await test('touch taps select up to three references; scroll gestures and cancellation do not',async()=>{
  const p=page();await p.load();enableExemplars(p);tap(p,100,100,{pointerType:'touch'});
  p.get('canvas').listeners.get('pointerdown')({clientX:150,clientY:100,button:0,pointerId:2,pointerType:'touch'});
  p.get('canvas').listeners.get('pointerup')({clientX:150,clientY:140,button:0,pointerId:2,pointerType:'touch',preventDefault(){}});
  p.get('canvas').listeners.get('pointerdown')({clientX:150,clientY:100,button:0,pointerId:2,pointerType:'touch'});
  p.events.get('pointercancel')(pointer(150,100,{pointerId:2,pointerType:'touch'}));p.get('canvas').listeners.get('pointerup')({clientX:150,clientY:100,button:0,pointerId:2,pointerType:'touch',preventDefault(){}});
  assert.equal(p.evaluate('exemplarPoints.length'),1);assert.equal(p.evaluate('originalHeld'),false);
  tap(p,200,200,{pointerType:'touch'});tap(p,300,300);tap(p,400,400);
  assert.equal(p.evaluate('exemplarPoints.length'),3);assert.match(p.get('exemplar-state').textContent,/3/);
  p.get('exemplar-undo').onclick();assert.equal(p.evaluate('exemplarPoints.length'),2);
  p.get('exemplar-clear').onclick();assert.equal(p.evaluate('exemplarPoints.length'),0);assert.equal(p.get('run-exemplars').disabled,true);
  await p.get('run-exemplars').onclick();assert.equal(p.state.fetchCalls,0);
 });
 await test('reference request carries visual examples; ordinary counting still sends only image',async()=>{
  const p=page();await p.load();enableExemplars(p);tap(p,123,234);
  range(p,32);
  p.fixture(result(5,{framework:'CountGD++',category:'bamboo'}));await p.get('run-exemplars').onclick();
  const payload=JSON.parse(p.calls[0].options.body);
  assert.deepEqual(payload,{image:'data:image/png;base64,AA',exemplar_points:[{x:123,y:234,diameter:32}],engine:'smart'});
  assert.equal(p.get('count').textContent,'5');assert.equal(p.evaluate('points.length'),5);assert.equal(p.evaluate('exemplarPoints.length'),1);
  assert.equal(p.get('result-card').dataset.state,'candidate');
  const commands=p.get('canvas').context.commands,labels=commands.filter(c=>c[0]==='fillText');assert(labels.some(label=>label[1]==='E1'));
  await p.run(result(2));assert.deepEqual(Object.keys(JSON.parse(p.calls[1].options.body)),['image']);assert.equal(p.evaluate('exemplarPoints.length'),1);
  p.get('exemplar-enabled').checked=false;p.get('exemplar-enabled').onchange();assert.equal(p.get('run-exemplars').disabled,true);
  await p.get('run-exemplars').onclick();assert.equal(p.state.fetchCalls,2);
 });
 await test('references remain separate from predictions and are locked during inference',async()=>{
  const p=page();await p.load();enableExemplars(p);tap(p,100,100);let deliver;
  p.state.response=()=>new Promise(resolve=>deliver=resolve);const running=p.get('run-exemplars').onclick();await p.flush();
  for(const id of ['exemplar-enabled','exemplar-size','exemplar-undo','exemplar-clear','run-exemplars','exemplar-delete','exemplar-smaller','exemplar-larger','exemplar-edit','zoom-in','zoom-out','zoom-reset'])assert.equal(p.get(id).disabled,true,id);
  tap(p,200,200);p.get('exemplar-clear').onclick();p.get('exemplar-undo').onclick();range(p,99);
  assert.equal(p.evaluate('exemplarPoints.length'),1);assert.equal(p.evaluate('exemplarDiameter'),25);
  deliver({ok:true,status:200,json:async()=>result(null,{prediction_valid:false,diagnosis:'no_final_detections'})});await running;
  assert.equal(p.get('count').textContent,'—');assert.equal(p.evaluate('points.length'),0);assert.equal(p.evaluate('exemplarPoints.length'),1);assert.equal(p.get('run-exemplars').disabled,false);
  await p.load();assert.equal(p.evaluate('exemplarPoints.length'),0);assert.equal(p.get('exemplar-enabled').checked,false);assert.equal(p.get('run-exemplars').disabled,true);
 });
 await test('reference counting keeps timeout behavior and ignores a stale result',async()=>{
  const p=page();await p.load();enableExemplars(p);tap(p,100,100);let deliver;
  p.state.response=()=>new Promise(resolve=>deliver=resolve);const running=p.get('run-exemplars').onclick();await p.flush();await p.advance(100000);await running;
  assert.equal(p.get('result-state').textContent,'等待超时');assert.equal(p.evaluate('exemplarPoints.length'),1);assert.equal(p.get('run-exemplars').disabled,false);assert.equal(p.timers.size,0);
  await p.load();deliver({ok:true,status:200,json:async()=>result(99)});await p.flush();
  assert.equal(p.get('count').textContent,'—');assert.equal(p.evaluate('points.length'),0);assert.equal(p.evaluate('exemplarPoints.length'),0);
 });
 await test('reference editing opens directly and selecting a target shows its crop',async()=>{
  const p=page();await p.load();enableExemplars(p);
  assert.equal(p.get('exemplar-editor').hidden,false);assert.equal(p.get('exemplar-inspector').hidden,false);
  assert.equal(p.get('exemplar-preview').hidden,true);tap(p,180,180);
  assert.equal(p.evaluate('selectedExemplar'),0);assert.equal(p.get('exemplar-preview').hidden,false);
  assert.equal(p.get('exemplar-empty').hidden,true);assert.equal(p.get('exemplar-card-0').getAttribute('aria-pressed'),'true');
  assert(p.get('exemplar-preview').context.commands.some(command=>command[0]==='drawImage'));
  assert(p.get('exemplar-thumb-0').context.commands.some(command=>command[0]==='drawImage'));
  p.get('exemplar-done').onclick();assert.equal(p.get('exemplar-enabled').checked,true);assert.equal(p.evaluate('editorCollapsed'),true);
  assert.equal(p.evaluate('exemplarPoints.length'),1);assert.equal(p.get('exemplar-editor').hidden,true);
 });
 await test('each example keeps its own size when selecting cards or adjusting the range',async()=>{
  const p=page();await p.load();enableExemplars(p);tap(p,100,100);range(p,40);tap(p,250,150);range(p,70);
  assert.equal(p.evaluate('exemplarPoints[0].diameter'),40);assert.equal(p.evaluate('exemplarPoints[1].diameter'),70);
  p.get('exemplar-card-0').onclick();assert.equal(p.evaluate('selectedExemplar'),0);assert.equal(Number(p.get('exemplar-size').value),40);
  p.get('exemplar-smaller').onclick();assert(p.evaluate('exemplarPoints[0].diameter')<40);assert.equal(p.evaluate('exemplarPoints[1].diameter'),70);
  p.get('exemplar-larger').onclick();assert(p.evaluate('exemplarPoints[0].diameter')>8);assert.equal(p.evaluate('exemplarPoints[1].diameter'),70);
  p.fixture(result(3));await p.get('run-exemplars').onclick();const payload=JSON.parse(p.calls[0].options.body);
  assert.equal(payload.exemplar_points[0].diameter,p.evaluate('exemplarPoints[0].diameter'));assert.equal(payload.exemplar_points[1].diameter,70);
 });
 await test('dragging a selected reference moves it without adding another target',async()=>{
  const p=page();await p.load();enableExemplars(p);tap(p,150,150);const size=p.evaluate('exemplarPoints[0].diameter');
  drag(p,[150,150],[220,190]);assert.equal(p.evaluate('exemplarPoints.length'),1);
  assert.equal(p.evaluate('exemplarPoints[0].x'),220);assert.equal(p.evaluate('exemplarPoints[0].y'),190);assert.equal(p.evaluate('exemplarPoints[0].diameter'),size);
  p.get('exemplar-undo').onclick();assert.equal(p.evaluate('exemplarPoints[0].x'),150);assert.equal(p.evaluate('exemplarPoints[0].y'),150);
 });
 await test('a small on-screen reference can still be moved from its centre',async()=>{
  const p=page();p.state.width=3000;p.state.height=2000;await p.load();enableExemplars(p);
  p.get('canvas').getBoundingClientRect=()=>({left:0,top:0,right:600,bottom:400,width:600,height:400});tap(p,300,200);range(p,40);
  drag(p,[300,200],[320,210]);assert.equal(p.evaluate('exemplarPoints[0].x'),1600);assert.equal(p.evaluate('exemplarPoints[0].y'),1050);
  assert.equal(p.evaluate('exemplarPoints[0].diameter'),40);assert.equal(p.evaluate('exemplarPoints.length'),1);
 });
 await test('corner dragging resizes around the centre and stays within supported limits',async()=>{
  const p=page();await p.load();enableExemplars(p);tap(p,200,200);range(p,80);
  drag(p,[240,240],[270,270]);assert.equal(p.evaluate('exemplarPoints[0].x'),200);assert.equal(p.evaluate('exemplarPoints[0].y'),200);
  assert(p.evaluate('exemplarPoints[0].diameter')>80);assert(p.evaluate('exemplarPoints[0].diameter')<=400);
  const diameter=p.evaluate('exemplarPoints[0].diameter');drag(p,[200+diameter/2,200+diameter/2],[900,900]);assert.equal(p.evaluate('exemplarPoints[0].diameter'),400);
  drag(p,[400,400],[201,201]);assert.equal(p.evaluate('exemplarPoints[0].diameter'),8);
 });
 await test('a dragged reference can finish outside the canvas and is clamped inside the image',async()=>{
  const p=page();await p.load();enableExemplars(p);tap(p,100,100);
  fire(p,'canvas','pointerdown',pointer(100,100));p.events.get('pointermove')?.(pointer(-50,-70));p.events.get('pointerup')?.(pointer(-50,-70));
  assert.equal(p.evaluate('exemplarPoints.length'),1);assert(p.evaluate('exemplarPoints[0].x')>=0);assert(p.evaluate('exemplarPoints[0].y')>=0);
  assert(p.evaluate('exemplarPoints[0].x')<100);assert.equal(p.evaluate('exemplarGesture'),null);
  p.get('exemplar-undo').onclick();assert.equal(p.evaluate('exemplarPoints[0].x'),100);assert.equal(p.evaluate('exemplarPoints[0].y'),100);
 });
 await test('pointer cancellation and Escape restore the reference before a drag',async()=>{
  const p=page();await p.load();enableExemplars(p);tap(p,180,180);range(p,70);const before=p.evaluate('JSON.stringify(exemplarPoints)');
  fire(p,'canvas','pointerdown',pointer(180,180));fire(p,'canvas','pointermove',pointer(240,210));p.events.get('pointercancel')(pointer(240,210));
  assert.equal(p.evaluate('JSON.stringify(exemplarPoints)'),before);assert.equal(p.evaluate('exemplarGesture'),null);
  fire(p,'canvas','pointerdown',pointer(215,215));fire(p,'canvas','pointermove',pointer(260,260));key(p,'Escape');
  assert.equal(p.evaluate('JSON.stringify(exemplarPoints)'),before);assert.equal(p.get('exemplar-enabled').checked,true);
  key(p,'Escape');assert.equal(p.get('exemplar-enabled').checked,true);assert.equal(p.evaluate('editorCollapsed'),true);
  assert.equal(p.get('exemplar-editor').hidden,true);assert.equal(p.get('exemplar-bottom').hidden,false);assert.equal(p.get('run-exemplars').disabled,false);
 });
 await test('one slider gesture is one undo step, and deletion and clearing can be undone',async()=>{
  const p=page();await p.load();enableExemplars(p);tap(p,160,160);const original=p.evaluate('exemplarPoints[0].diameter');
  range(p,45,false);range(p,65,false);range(p,90,true);assert.equal(p.evaluate('exemplarPoints[0].diameter'),90);
  p.get('exemplar-undo').onclick();assert.equal(p.evaluate('exemplarPoints[0].diameter'),original);assert.equal(p.evaluate('exemplarPoints.length'),1);
  p.get('exemplar-delete').onclick();assert.equal(p.evaluate('exemplarPoints.length'),0);p.get('exemplar-undo').onclick();assert.equal(p.evaluate('exemplarPoints.length'),1);
  tap(p,350,240);p.get('exemplar-clear').onclick();assert.equal(p.evaluate('exemplarPoints.length'),0);p.get('exemplar-undo').onclick();assert.equal(p.evaluate('exemplarPoints.length'),2);
  await p.load();assert.equal(p.evaluate('selectedExemplar'),-1);assert.equal(p.get('exemplar-undo').disabled,true);
 });
 await test('zoom affects only the view and native image coordinates still map correctly',async()=>{
  const p=page();p.state.width=1600;p.state.height=1000;await p.load();enableExemplars(p);tap(p,200,200);
  const before=p.evaluate('JSON.stringify(exemplarPoints)'),url=p.evaluate('dataURL');p.get('zoom-in').onclick();
  assert.notEqual(p.get('zoom-label').textContent,'100%');assert.equal(p.evaluate('JSON.stringify(exemplarPoints)'),before);assert.equal(p.evaluate('dataURL'),url);
  p.get('canvas').getBoundingClientRect=()=>({left:-200,top:-100,right:1400,bottom:900,width:1600,height:1000});tap(p,600,400);
  assert.equal(p.evaluate('exemplarPoints[1].x'),800);assert.equal(p.evaluate('exemplarPoints[1].y'),500);
  p.get('zoom-reset').onclick();assert.equal(p.get('zoom-label').textContent,'100%');assert.equal(p.evaluate('dataURL'),url);
  for(let i=0;i<20;i++)p.get('zoom-in').onclick();assert.equal(p.get('zoom-label').textContent,'400%');
  for(let i=0;i<30;i++)p.get('zoom-out').onclick();assert.equal(p.get('zoom-label').textContent,'100%');
 });
 await test('keyboard deletion and undo work without stealing keys from form controls',async()=>{
  const p=page();await p.load();enableExemplars(p);tap(p,150,150);tap(p,300,220);
  key(p,'Delete',{target:p.get('file')});assert.equal(p.evaluate('exemplarPoints.length'),2);
  key(p,'Delete');assert.equal(p.evaluate('exemplarPoints.length'),1);key(p,'z',{ctrlKey:true});assert.equal(p.evaluate('exemplarPoints.length'),2);
  p.fixture(result(6));key(p,'Enter',{target:p.get('exemplar-size')});await p.flush();assert.equal(p.state.fetchCalls,0);
  key(p,'Enter');await p.flush();assert.equal(p.state.fetchCalls,1);assert.equal(p.get('count').textContent,'6');
 });
 await test('changing references marks existing results for recount without changing the count',async()=>{
  const p=page();await p.load();await p.run(result(7));enableExemplars(p);tap(p,190,190);
  assert.equal(p.get('count').textContent,'7');assert.equal(p.get('exemplar-dirty').hidden,false);
  p.fixture(result(8));await p.get('run-exemplars').onclick();assert.equal(p.get('count').textContent,'8');assert.equal(p.get('exemplar-dirty').hidden,true);
  range(p,65);assert.equal(p.get('count').textContent,'8');assert.equal(p.get('exemplar-dirty').hidden,false);
 });
 await test('collapsing adjustments preserves the recount action and never adds accidental examples',async()=>{
  const p=page();await p.load();await p.run(result(7));enableExemplars(p);tap(p,170,180);range(p,60);const selected=p.evaluate('JSON.stringify(exemplarPoints)');
  p.get('exemplar-done').onclick();assert.equal(p.get('exemplar-enabled').checked,true);assert.equal(p.evaluate('editorCollapsed'),true);
  assert.equal(p.get('exemplar-inspector').hidden,true);assert.equal(p.get('exemplar-bottom').hidden,false);assert.equal(p.get('exemplar-edit').hidden,false);
  assert.equal(p.get('run-exemplars').disabled,false);assert.equal(p.get('exemplar-dirty').hidden,false);
  tap(p,340,260);range(p,100);key(p,'Delete');assert.equal(p.evaluate('JSON.stringify(exemplarPoints)'),selected);assert.equal(p.evaluate('originalHeld'),false);
  p.fixture(result(8));await p.get('run-exemplars').onclick();assert.equal(p.get('count').textContent,'8');assert.equal(p.get('exemplar-dirty').hidden,true);
  p.get('exemplar-edit').onclick();assert.equal(p.evaluate('editorCollapsed'),false);assert.equal(p.get('exemplar-inspector').hidden,false);
  assert.equal(p.evaluate('JSON.stringify(exemplarPoints)'),selected);assert.equal(Number(p.get('exemplar-size').value),60);
 });
 await test('collapsed adjustments cannot reopen or change while recounting',async()=>{
  const p=page();await p.load();enableExemplars(p);tap(p,160,170);p.get('exemplar-done').onclick();let deliver;
  p.state.response=()=>new Promise(resolve=>deliver=resolve);const running=p.get('run-exemplars').onclick();await p.flush();
  assert.equal(p.get('exemplar-edit').disabled,true);assert.equal(p.get('run-exemplars').disabled,true);const selected=p.evaluate('JSON.stringify(exemplarPoints)');
  p.get('exemplar-edit').onclick();tap(p,300,300);assert.equal(p.evaluate('editorCollapsed'),true);assert.equal(p.evaluate('JSON.stringify(exemplarPoints)'),selected);
  deliver({ok:true,status:200,json:async()=>result(6)});await running;assert.equal(p.get('exemplar-edit').disabled,false);assert.equal(p.get('run-exemplars').disabled,false);
  p.get('exemplar-edit').onclick();assert.equal(p.evaluate('editorCollapsed'),false);assert.equal(p.evaluate('JSON.stringify(exemplarPoints)'),selected);
 });
 await test('resizing a reference clipped by the image edge uses drag distance without jumping',async()=>{
  const p=page();await p.load();enableExemplars(p);tap(p,10,10);range(p,100);
  drag(p,[0,0],[5,5]);assert.equal(p.evaluate('exemplarPoints[0].diameter'),90);
  assert.equal(p.evaluate('exemplarPoints[0].x'),10);assert.equal(p.evaluate('exemplarPoints[0].y'),10);
  p.get('exemplar-undo').onclick();assert.equal(p.evaluate('exemplarPoints[0].diameter'),100);
 });
 await test('opening the narrower inspector clamps the pan while keeping reference coordinates',async()=>{
  const p=page();await p.load();
  p.get('canvas').getBoundingClientRect=()=>{const base=p.get('editing-workspace').dataset.editing==='true'?400:800,z=p.evaluate('viewZoom'),width=base*z,height=base*500/800*z;return{left:0,top:0,width,height,right:width,bottom:height}};
  p.get('zoom-in').onclick();p.get('zoom-in').onclick();p.get('pan-mode').onclick();drag(p,[400,250],[780,250]);
  assert.equal(p.evaluate('viewZoom'),2);assert(p.evaluate('viewPan.x')>200);
  enableExemplars(p);assert(p.evaluate('viewPan.x')<=200);assert(p.evaluate('viewPan.x')>=-200);
  tap(p,400,250);const selected=p.evaluate('JSON.stringify(exemplarPoints)');
  p.get('exemplar-done').onclick();assert.equal(p.evaluate('JSON.stringify(exemplarPoints)'),selected);
  p.get('exemplar-edit').onclick();assert(p.evaluate('viewPan.x')<=200);assert.equal(p.evaluate('JSON.stringify(exemplarPoints)'),selected);
 });
 console.log('34 page behavior checks passed; no browser or GPU used.');
})().catch(error=>{console.error(error);process.exitCode=1});
