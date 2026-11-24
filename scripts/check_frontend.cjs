// Compile the real Vue template without starting a browser or a model request.
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const root=path.resolve(__dirname,'..');
const context=vm.createContext({console});
vm.runInContext(fs.readFileSync(path.join(root,'frontend/assets/vue.global.prod.js'),'utf8'),context);
const html=fs.readFileSync(path.join(root,'frontend/index.html'),'utf8');
const template=html.match(/<body[^>]*>([\s\S]*?)<script/)[1];
context.Vue.compile(template,{decodeEntities:text=>text,onError:error=>{throw error;}});
console.log('Vue template compiled');
