// Canvas API contract smoke test only; not a browser screenshot/layout test.
const fs=require('fs'), vm=require('vm'), path=require('path');
const file=process.argv[2]||path.join(__dirname,'../output/ur5e/ur5e_cleaning_3d.html');
const html=fs.readFileSync(file,'utf8'),script=html.match(/<script>([\s\S]*?)<\/script>/)[1];
let commands=0;
const ctx=new Proxy({}, {get:(o,key)=>o[key]??((...args)=>{for(const a of args)if(typeof a==='number'&&!Number.isFinite(a))throw Error('Non-finite canvas coordinate');commands++;}),set:(o,k,v)=>(o[k]=v,true)});
const elements={};
const element=id=>elements[id]??(elements[id]={value:0,textContent:'',insertAdjacentHTML(){},getContext:()=>ctx,getBoundingClientRect:()=>({width:1000,height:650}),setPointerCapture(){}});
const sandbox={document:{getElementById:element,querySelector:element,querySelectorAll:()=>Array.from({length:5},(_,i)=>element('note'+i))},devicePixelRatio:1,setInterval(){},window:{},console};
vm.createContext(sandbox);
vm.runInContext(script,sandbox);
vm.runInContext('index=Math.floor(data.frames.length/2);render();index=data.frames.length-1;render();',sandbox);
if(!element('pose').textContent.includes('J6'))throw Error('Missing joint display');
console.log(JSON.stringify({status:'passed',frames_checked:3,canvas_commands:commands,scope:'mock canvas runtime, not browser visual verification'}));
