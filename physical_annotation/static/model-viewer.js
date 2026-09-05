import {context, api, flush, message} from './app.js';
const panel=document.getElementById('model-panel');
let blind=true, requestGeneration=0;
const warning='Viewing model outputs before verifying ground truth may introduce reviewer bias.';
function node(tag,text,className){const element=document.createElement(tag);if(text!==undefined)element.textContent=text;if(className)element.className=className;return element;}

function render(){
  requestGeneration++;
  panel.replaceChildren();panel.className='panel';
  panel.append(node('h2','Existing Model Outputs'));
  const label=node('label',undefined,'check');const toggle=node('input');toggle.type='checkbox';toggle.id='blind-mode';toggle.checked=blind;
  label.append(toggle,document.createTextNode(`Blind annotation mode: ${blind?'ON':'OFF'}`));panel.append(label);
  panel.append(node('p',warning,'notice'));
  const locked=blind&&!context.current.human_verified;
  const hint=node('p',locked?'Annotate ground truth first. Verify this image to unlock the read-only comparison.':'Model outputs remain hidden until you click Show model outputs.','muted');panel.append(hint);
  const show=node('button','Show model outputs');show.id='show-model-outputs';show.disabled=locked;panel.append(show);
  const results=node('div');results.id='model-output-results';results.hidden=true;panel.append(results);
  toggle.onchange=()=>{
    if(!toggle.checked&&!window.confirm(`${warning}\nTurn blind annotation mode OFF for this browser session?`)){toggle.checked=true;return;}
    blind=toggle.checked;render();
  };
  show.onclick=async()=>{
    const requestedImageId=context.current.image_id, requestedBlind=blind;
    show.disabled=true;
    try{
      await flush();
      if(context.current.image_id!==requestedImageId||blind!==requestedBlind)return;
      const liveResults=document.getElementById('model-output-results');
      const liveShow=document.getElementById('show-model-outputs');
      const generation=requestGeneration, imageId=context.current.image_id;
      const data=await api(`/api/model-outputs/${encodeURIComponent(imageId)}`,{show:true,blind_mode:blind});
      if(generation!==requestGeneration||context.current.image_id!==imageId)return;
      liveResults.replaceChildren();
      for(const output of data.outputs){
        const card=node('section',undefined,'region-card');card.append(node('h3',output.model_name));
        card.append(node('p',`Action: ${output.action===null?'No parsed action preserved':output.action}`));
        card.append(node('pre',`${output.critical_argument_name} = ${JSON.stringify(output.critical_argument)}`));
        card.append(node('p',`Completed: ${output.completed} · Parse valid: ${output.parse_valid} · Schema valid: ${output.schema_valid}`,'muted'));
        if(output.error_type)card.append(node('p',`Preserved error: ${output.error_type}`,'notice'));
        const details=node('details');details.append(node('summary','Preserved output text and source hashes'));
        details.append(node('pre',output.output_text??'(No output text preserved)'));details.append(node('pre',JSON.stringify(output.provenance,null,2)));card.append(details);liveResults.append(card);
      }
      liveResults.hidden=false;liveShow.textContent='Hide model outputs';liveShow.disabled=false;liveShow.onclick=render;
    }catch(error){message(error.message,true);show.disabled=false;}
  };
}
document.addEventListener('annotation-render',render);
document.addEventListener('annotation-edited',render);
document.addEventListener('annotation-saved',render);
