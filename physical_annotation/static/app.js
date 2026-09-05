// Human decisions only. Requests are confined to the local annotation server.
const $ = id => document.getElementById(id);
export const context = {state:null, current:null, dirty:false, busy:false, filter:'ALL', generation:0};
let timer, writes = Promise.resolve();
export function message(text, error=false) { $('save-status').textContent=text; $('save-status').classList.toggle('error',error); }
export function setBusy(busy){
  context.busy=busy;document.querySelector('.workspace').inert=busy;
  for(const id of ['previous','save','needs-review','verify','next'])$(id).disabled=busy;
}
export async function api(path, body) {
  const response=await fetch(path, body === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json','X-Annotation-Token':context.state.token},body:JSON.stringify(body)});
  const result=await response.json();
  if (!response.ok) throw new Error(typeof result.detail==='string' ? result.detail : JSON.stringify(result.detail));
  return result;
}
export function edit() {
  context.dirty=true; context.generation++;
  context.current.human_verified=false;
  context.current.reviewed_at=null;
  if(context.current.status!=='NEEDS_REVIEW') context.current.status='DRAFT';
  for(const region of context.current.regions||[]) region.human_verified=false;
  $('status').textContent=context.current.status; $('status').className=`badge ${context.current.status}`;
  message('Unsaved changes…'); clearTimeout(timer); timer=setTimeout(()=>save().catch(error=>message(error.message,true)),650);
  document.dispatchEvent(new CustomEvent('annotation-edited'));
}
export function save(verify=false, confirmAttackerMatch=false) {
  clearTimeout(timer);
  const run=async()=>{
    if(!context.dirty && !verify) return;
    const annotation=structuredClone(context.current), generation=context.generation;
    message(verify?'Verifying…':'Saving draft…');
    const result=await api(`/api/annotations/${encodeURIComponent(annotation.image_id)}`,{annotation,expected_revision:context.state.revision,verify,reviewer:$('reviewer').value.trim()||null,confirm_attacker_match:confirmAttackerMatch});
    context.state={...result,token:context.state.token};
    if(generation===context.generation){context.current=structuredClone(result.annotations.find(a=>a.image_id===annotation.image_id));context.dirty=false;}
    renderNavigation(); renderStatus(); message(context.dirty?'New changes pending…':verify?'Human verification saved':'Draft saved');
    document.dispatchEvent(new CustomEvent('annotation-saved'));
  };
  const pending=writes.then(run); writes=pending.catch(()=>{}); return pending;
}
function filtered() {
  const query=$('search').value.toLowerCase();
  return context.state.annotations.filter(a=>(context.filter==='ALL'||a.status===context.filter||a.scenario===context.filter||(context.filter==='RISK'&&a.inference_contamination_risk))&&a.original_filename.toLowerCase().includes(query));
}
export async function flush(){do {await save();} while(context.dirty);}
export async function jump(id) {
  if(context.busy)return;
  setBusy(true);
  try{
    await flush();
    context.current=structuredClone(context.state.annotations.find(a=>a.image_id===id));
    localStorage.setItem('physical-annotation-current',id); context.generation++; render();
  }finally{setBusy(false);}
}
export async function navigate(offset) {
  const rows=filtered(), index=rows.findIndex(a=>a.image_id===context.current.image_id);
  const next=rows[Math.max(0,Math.min(rows.length-1,index+offset))]; if(next) await jump(next.image_id);
}
function renderNavigation() {
  const rows=context.state.annotations;
  const verified=rows.filter(a=>a.status==='VERIFIED').length;
  $('progress').max=rows.length; $('progress').value=verified;
  $('position').textContent=`${rows.findIndex(a=>a.image_id===context.current.image_id)+1} / ${rows.length} · ${Math.round(100*verified/rows.length)}% verified`;
  $('dashboard').replaceChildren();
  for(const [name, count] of [['Total',rows.length],...['VERIFIED','NEEDS_REVIEW','DRAFT','UNREVIEWED'].map(s=>[s.replaceAll('_',' '),rows.filter(a=>a.status===s).length])]){
    const line=document.createElement('div');line.textContent=`${name}: ${count}`;$('dashboard').append(line);
  }
  for(const scenario of ['CALL','RESTAURANT_RESERVATION','NAVIGATION','SAFETY']){
    const group=rows.filter(a=>a.scenario===scenario),line=document.createElement('div');line.textContent=`${scenario.replace('_RESERVATION','')}: ${group.filter(a=>a.human_verified).length} / ${group.length}`;$('dashboard').append(line);
  }
  $('filters').replaceChildren();
  for(const [value,label] of [['ALL','All images'],['UNREVIEWED','Unreviewed'],['VERIFIED','Verified'],['NEEDS_REVIEW','Needs Review'],['DRAFT','Draft'],['CALL','CALL'],['RESTAURANT_RESERVATION','Restaurant'],['NAVIGATION','Navigation'],['SAFETY','Safety'],['RISK','Contaminated-risk images']]){
    const button=document.createElement('button');const count=rows.filter(a=>value==='ALL'||a.status===value||a.scenario===value||(value==='RISK'&&a.inference_contamination_risk)).length;button.textContent=`${label}  ${count}`;button.classList.toggle('active',context.filter===value);button.onclick=()=>{context.filter=value;renderNavigation();};$('filters').append(button);
  }
  $('image-list').replaceChildren();
  for(const a of filtered()){
    const button=document.createElement('button');button.className='image-item';button.classList.toggle('active',a.image_id===context.current.image_id);button.textContent=`${a.human_verified?'✓':a.status==='NEEDS_REVIEW'?'⚠':'○'} ${a.original_filename}`;const detail=document.createElement('small');detail.textContent=`${a.scenario} · ${a.status}`;button.append(detail);button.onclick=()=>jump(a.image_id).catch(e=>message(e.message,true));$('image-list').append(button);
  }
}
function renderStatus(){const a=context.current;$('status').textContent=a.status;$('status').className=`badge ${a.status}`;$('prefill-warning').textContent=a.human_verified?`HUMAN VERIFIED · ${a.reviewer} · ${a.reviewed_at}`:'PRE-FILLED / DRAFT — NOT HUMAN VERIFIED';}
function render(){
  const a=context.current;renderNavigation();renderStatus();$('filename').textContent=a.original_filename;$('image').src=`/api/images/${encodeURIComponent(a.image_id)}`;
  $('image-error').hidden=true;$('image').onerror=()=>{$('image-error').hidden=false;};
  $('notes').value=a.notes||'';$('risk').checked=a.inference_contamination_risk;$('exclude').checked=a.exclude_from_primary_aggregate;
  $('exclusion-reason').value=a.exclusion_reason||'';
  $('reviewer').value=a.reviewer||localStorage.getItem('physical-annotation-reviewer')||'';
  $('contamination-note').hidden=!a.inference_contamination_risk;$('contamination-note').textContent=a.contamination_note||'Review contamination risk and decide aggregate inclusion explicitly.';
  $('source-notes').textContent=JSON.stringify(a.prefill||{},null,2);
  $('scenario-form').textContent=`Scenario: ${a.scenario} · Attack mode: ${a.attack_mode}`;
  document.dispatchEvent(new CustomEvent('annotation-render'));
}
for(const [id,key,type] of [['notes','notes','text'],['risk','inference_contamination_risk','bool'],['exclude','exclude_from_primary_aggregate','bool'],['exclusion-reason','exclusion_reason','text']]){
  $(id).addEventListener('input',()=>{context.current[key]=type==='bool'?$(id).checked:$(id).value;edit();});
}
$('reviewer').oninput=()=>{localStorage.setItem('physical-annotation-reviewer',$('reviewer').value);};
$('search').oninput=renderNavigation;
$('save').onclick=()=>save().catch(e=>message(e.message,true));
$('previous').onclick=()=>navigate(-1).catch(e=>message(e.message,true));
$('next').onclick=()=>navigate(1).catch(e=>message(e.message,true));
$('needs-review').onclick=()=>{context.current.ground_truth_known=false;context.current.ground_truth_value=null;context.current.status='NEEDS_REVIEW';edit();document.dispatchEvent(new CustomEvent('annotation-render'));};
$('verify').disabled=true; // Scenario-specific verification is connected in the next checkpoint.
document.addEventListener('keydown',event=>{
  if(['INPUT','TEXTAREA','SELECT'].includes(event.target.tagName)||event.target.isContentEditable||document.querySelector('dialog[open]'))return;
  if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==='s'){event.preventDefault();$('save').click();}
  else if(!event.ctrlKey&&!event.metaKey&&!event.altKey){if(event.key==='ArrowLeft')$('previous').click();if(event.key==='ArrowRight')$('next').click();if(event.key.toLowerCase()==='v')$('verify').click();if(event.key.toLowerCase()==='n')$('needs-review').click();}
});
window.addEventListener('beforeunload',event=>{if(context.dirty){event.preventDefault();event.returnValue='';}});
async function start(){await import('./forms.js');await import('./regions.js');await import('./model-viewer.js');context.state=await api('/api/state');const remembered=localStorage.getItem('physical-annotation-current');context.current=structuredClone(context.state.annotations.find(a=>a.image_id===remembered)||context.state.annotations[0]);render();}
start().catch(e=>message(e.message,true));
