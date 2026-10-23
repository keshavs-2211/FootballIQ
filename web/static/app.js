(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const state = {projects:[], current:null, selectedFile:null, crop:null, role:'team_a',
    cropDrag:null, landmarks:[], marked:[], poll:null, frameTime:0, calibTime:0, view:'overview'};
  const roleNames = {
    team_a: 'Team A',
    team_b: 'Team B',
    team_a_gk: 'Team A goalkeeper',
    team_b_gk: 'Team B goalkeeper',
    referee: 'Referee',
    ball: 'Ball'
  };
  const requiredRoles = ['referee'];
  const optionalRoles = ['team_a','team_b','team_a_gk','team_b_gk'];
  const escapeHtml = text => String(text??'').replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const fmt = v=>Number.isFinite(Number(v))?Number(v).toLocaleString(undefined,{maximumFractionDigits:1}):'—';
  const duration = s => {let n=Math.max(0,Number(s)||0);return `${Math.floor(n/60).toString().padStart(2,'0')}:${(n%60).toFixed(1).padStart(4,'0')}`;};
  const notify = (message, isError=false)=>{const el=$('toast');el.textContent=message;el.classList.toggle('error',isError);el.classList.add('show');clearTimeout(notify.timer);notify.timer=setTimeout(()=>el.classList.remove('show'),4300);};
  async function api(url, opts={}){const res=await fetch(url,opts);let body={};try{body=await res.json()}catch{}if(!res.ok)throw new Error(body.error||body.description||`Request failed: ${res.status}`);return body;}
  const jsonOpts = obj => ({method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(obj)});
  function nav(view){clearTimeout(state.videoPoll);state.view=view;document.querySelectorAll('.view').forEach(e=>e.classList.toggle('active',e.id===`view-${view}`));document.querySelectorAll('.nav-link').forEach(e=>e.classList.toggle('active',e.dataset.nav===view));$('current-page-label').textContent={overview:'Overview',setup:'Match studio',analytics:'Analytics',library:'Match library'}[view];window.scrollTo({top:0,behavior:'smooth'});if(view==='analytics')renderAnalytics();else if(view==='overview')refreshOverviewVideos();}
  function switchSetup(part){['upload','config','processing'].forEach(p=>$(`setup-${p}`).classList.toggle('hidden',p!==part));let step=part==='upload'?1:part==='config'?2:3;[1,2,3].forEach(i=>{const el=$(`step-dot-${i}`);el.classList.toggle('active',i===step);el.classList.toggle('complete',i<step)});}
  function projectCard(p){let status=p.state==='done'?'Analyzed':p.state==='running'?'Processing':p.state==='failed'?'Needs attention':'Setup needed';return `<button class="project-card" data-project="${p.id}"><div class="project-card-top"><div class="project-card-icon">◈</div><span class="state-label ${p.state}">${status}</span></div><h3>${escapeHtml(p.title)}</h3><p>${escapeHtml(p.team_a)} <span style="color:var(--accent)">vs</span> ${escapeHtml(p.team_b)}</p><div class="project-card-bottom"><span>${fmt(p.meta.duration_s)} sec · ${p.meta.width}×${p.meta.height}</span><span>Explore ↗</span></div></button>`;}
  function refreshLists(){const projects=state.projects;const empty='<div class="empty-block">No matches yet. Upload a video to begin your first analysis.</div>';$('hero-count').textContent=String(projects.length).padStart(2,'0');$('overview-projects').innerHTML=projects.length?projects.slice(0,3).map(projectCard).join(''):empty;$('library-projects').innerHTML=projects.length?projects.map(projectCard).join(''):empty;$('sidebar-projects').innerHTML=projects.length?projects.slice(0,5).map(p=>`<button class="side-project" data-project="${p.id}"><span class="side-project-mark"></span><div><strong>${escapeHtml(p.title)}</strong><small>${p.state==='done'?'Analysis complete':p.state==='running'?'Processing…':'Match setup'}</small></div></button>`).join(''):'<div class="side-empty">Your uploaded matches appear here.</div>';}
  async function loadProjects(){try{state.projects=(await api('/api/projects')).projects;refreshLists();refreshOverviewVideos();}catch(e){notify(e.message,true)}}
  async function refreshOverviewVideos(){
    const spot=$('overview-video-spotlight');
    if(!spot)return;
    const latest=(state.projects||[]).find(x=>x.state==='done');
    if(!latest){spot.innerHTML='<div class="empty-block">Complete or import an analysis to preview your three match videos here.</div>';return;}
    try{
      const data=await api(`/api/projects/${latest.id}/results`);
      const videos=(data.files||[]).filter(f=>f.kind==='video');
      const find=role=>videos.find(f=>f.role===role);
      const intro=`<div class="overview-video-title"><strong>${escapeHtml(latest.title)}</strong><span>${escapeHtml(latest.team_a)} vs ${escapeHtml(latest.team_b)}</span></div>`;
      spot.innerHTML=intro+(data.media_state==='running'?'<p class="video-prep-note">Converting videos for browser playback…</p>':'')+`<div class="match-video-grid">${videoCard(find('annotated'),'annotated','Annotated footage')}${videoCard(find('birdseye'),'birdseye',"Bird's-eye view")}${videoCard(find('side_by_side'),'side_by_side','Side-by-side comparison')}</div>`;
      clearTimeout(state.overviewMediaPoll);
      if(data.media_state==='running')state.overviewMediaPoll=setTimeout(refreshOverviewVideos,3000);
    }catch(err){spot.innerHTML=`<div class="empty-block">Could not load videos: ${escapeHtml(err.message)}</div>`;}
  }

  function handleProject(p){state.current=p;state.marked=[];state.crop=null;if(p.state==='running'){nav('setup');switchSetup('processing');startPolling(p.id);return;}if(p.state==='done'){nav('analytics');return;}nav('setup');switchSetup('config');initConfig();}
  async function openProject(id){try{const d=await api(`/api/projects/${id}`);handleProject(d.project)}catch(e){notify(e.message,true)}}
  function getTheme(){return localStorage.getItem('footballiq_theme')||'dark'}
  function setTheme(theme){document.documentElement.dataset.theme=theme;$('theme-icon').textContent=theme==='dark'?'☼':'☾';localStorage.setItem('footballiq_theme',theme)}
  function setupUpload(){nav('setup');switchSetup('upload');$('video-file').value='';state.selectedFile=null;$('selected-file').classList.add('hidden');loadLocalMatches()}
  async function loadLocalMatches(){
    try{
      const matches=(await api('/api/local-matches')).matches||[];
      $('local-match-select').innerHTML='<option value="">Select an existing match…</option>'+matches.map(m=>`<option value="${escapeHtml(m.clip)}">${escapeHtml(m.video)} ${m.has_outputs?'· Outputs found':'· Video only'}</option>`).join('');
      $('import-existing').disabled=!matches.length;
    }catch(e){$('local-match-select').innerHTML='<option value="">Could not load existing inputs</option>'}
  }
  async function importExisting(){
    let clip=$('local-match-select').value;
    if(!clip){notify('Choose an existing match.',true);return;}
    const btn=$('import-existing');btn.disabled=true;
    try{
      const data=await api('/api/projects/import',jsonOpts({clip}));
      await loadProjects();handleProject(data.project);
      notify(data.project.state==='done'?'Existing outputs opened. Videos will be prepared for browser playback.':'Existing video imported. Configure jerseys and calibration before processing.');
    }catch(e){notify(e.message,true)}finally{btn.disabled=false}
  }
  function chooseFile(file){if(!file)return;if(!file.name.toLowerCase().endsWith('.mp4')){notify('Please select an MP4 file.',true);return}state.selectedFile=file;$('selected-file').textContent=`◈ ${file.name} · ${(file.size/(1024*1024)).toFixed(1)} MB`;$('selected-file').classList.remove('hidden');if(!$('match-name').value)$('match-name').value=file.name.replace(/\.mp4$/i,'').replace(/_/g,' ')}
  async function submitVideo(){if(!state.selectedFile){notify('Select a match video first.',true);return}const btn=$('upload-submit');btn.disabled=true;btn.textContent='Uploading footage…';try{const fd=new FormData();fd.append('video',state.selectedFile);fd.append('title',$('match-name').value.trim());fd.append('team_a',$('team-a-name').value.trim());fd.append('team_b',$('team-b-name').value.trim());const d=await api('/api/projects',{method:'POST',body:fd});await loadProjects();handleProject(d.project);notify('Match uploaded. Now select your team jerseys and pitch landmarks.')}catch(e){notify(e.message,true)}finally{btn.disabled=false;btn.innerHTML='Continue to setup <span>→</span>'}}
  function setFrame(time){if(!state.current)return;let p=state.current;let max=Math.max(0,p.meta.duration_s-0.1);state.frameTime=Math.min(max,Math.max(0,Number(time)));$('frame-time').value=state.frameTime;$('frame-time-value').textContent=duration(state.frameTime);$('config-frame').src=`/api/projects/${p.id}/frame?time=${state.frameTime.toFixed(2)}&v=${Date.now()}`;state.crop=null;drawCrop();}
  function setCalibFrame(seconds, resetPoints=true){
    const p=state.current;if(!p)return;
    const fps=Number(p.meta.fps)||30, maxFrame=Math.max(0,p.meta.frames-1);
    const frame=Math.max(0,Math.min(maxFrame,Math.round((Number(seconds)||0)*fps)));
    const t=frame/fps;
    if(resetPoints && Math.abs(t-state.calibTime)>0.25/fps && state.marked.length){
      state.marked=[];
      notify('Calibration frame changed: unsaved landmarks were cleared.');
    }
    state.calibTime=t;
    $('calib-time').value=String(t);
    $('calib-time-input').value=t.toFixed(3);
    $('calib-time-value').textContent=duration(t)+` · Frame ${frame}`;
    $('calib-frame').src=`/api/projects/${p.id}/frame?time=${t.toFixed(5)}&v=${Date.now()}`;
    drawCalibration();
  }
  function initConfig(){
    let p=state.current;if(!p)return;
    $('setup-match-tag').textContent=p.team_a+' vs '+p.team_b;
    const max=Math.max(0,(p.meta.frames-1)/p.meta.fps), step=1/p.meta.fps;
    for(const id of ['frame-time','calib-time']){
      $(id).max=String(max);$(id).step=String(step);
    }
    $('calib-time-input').max=String(max);
    $('calib-time-input').step=String(step);
    $('frame-time').value='0';setFrame(0);
    state.marked=[];
    setCalibFrame(p.calibration_time_s||0,false);
    showReferences();
  }
  function showReferences(){
    let p=state.current;if(!p)return;
    for(const role of Object.keys(roleNames)){
      const samples=p.reference_samples?.[role]||[];
      const count=samples.filter(Boolean).length;
      const required=requiredRoles.includes(role), total=3;
      let status=$(`status-${role}`);
      status.textContent=`${count} / 3${count===3?' ✓':''}`;
      status.classList.toggle('ok',required?count===3:count>0);
      const first=samples.findIndex(Boolean)+1;
      $(`thumb-${role}`).style.backgroundImage=first?`url(/api/projects/${p.id}/reference/${role}?sample=${first}&v=${Date.now()})`:'';
    }
    const ready=requiredRoles.every(role=>(p.reference_samples?.[role]||[]).filter(Boolean).length===3);
    p.references_ready=ready;
    const role=state.role,samples=p.reference_samples?.[role]||[];
    const num=samples.filter(Boolean).length;
    $('reference-active-status').textContent=`${roleNames[role]}: ${num} / 3 photos (${role==='referee'?'required':role==='ball'?'required with ball analysis':'optional'})`;
    $('reference-preview').innerHTML=Array.from({length:3},(_,i)=>{
      const exists=Boolean(samples[i]);
      return `<button class="preview-slot ${exists?'ready':''}" type="button" data-ref-slot="${i+1}" aria-label="Select reference photo ${i+1}">${exists?`<img src="/api/projects/${p.id}/reference/${role}?sample=${i+1}&v=${Date.now()}" alt="${roleNames[role]} reference ${i+1}"/>`:'<span>+</span>'}<small>PHOTO ${i+1} ${exists?'✓':''}</small></button>`;
    }).join('');
    $('calib-badge').textContent=p.calibrated?'Calibrated ✓':'Not calibrated';
    $('calib-badge').classList.toggle('ok',p.calibrated);
    $('calib-badge').classList.toggle('pending',!p.calibrated);
  }
  function canvasInit(canvas,img){if(!img.complete||!img.naturalWidth)return null;const rect=img.getBoundingClientRect();let pixelRatio=Math.min(window.devicePixelRatio||1,2);canvas.width=Math.max(1,Math.round(rect.width*pixelRatio));canvas.height=Math.max(1,Math.round(rect.height*pixelRatio));let ctx=canvas.getContext('2d');ctx.scale(canvas.width/rect.width,canvas.height/rect.height);return {ctx,w:rect.width,h:rect.height};}
  function drawCrop(){const c=$('crop-canvas'),im=$('config-frame');const v=canvasInit(c,im);if(!v)return;const {ctx,w,h}=v;ctx.clearRect(0,0,w,h);if(!state.crop)return;let [x1,y1,x2,y2]=state.crop;ctx.fillStyle='#d1fe6224';ctx.fillRect(x1*w,y1*h,(x2-x1)*w,(y2-y1)*h);ctx.strokeStyle='#d1fe62';ctx.lineWidth=2;ctx.setLineDash([6,4]);ctx.strokeRect(x1*w,y1*h,(x2-x1)*w,(y2-y1)*h);ctx.setLineDash([]);}
  function canvasUV(event,canvas){const r=canvas.getBoundingClientRect();return [Math.max(0,Math.min(1,(event.clientX-r.left)/r.width)),Math.max(0,Math.min(1,(event.clientY-r.top)/r.height))]}
  function attachCrop(){const canvas=$('crop-canvas');canvas.addEventListener('pointerdown',e=>{canvas.setPointerCapture(e.pointerId);state.cropDrag=canvasUV(e,canvas);state.crop=null});canvas.addEventListener('pointermove',e=>{if(!state.cropDrag)return;let uv=canvasUV(e,canvas),p=state.cropDrag;state.crop=[Math.min(p[0],uv[0]),Math.min(p[1],uv[1]),Math.max(p[0],uv[0]),Math.max(p[1],uv[1])];drawCrop()});canvas.addEventListener('pointerup',()=>{state.cropDrag=null;$('crop-hint').textContent=state.crop?'Crop selected':'No selection'})}
  async function saveReference(){
    if(!state.current){notify('Upload a video first',true);return}
    if(!state.crop){notify('Draw a crop around the jersey, referee shirt or ball.',true);return}
    try{
      const id=state.current.id;
      const sample=Number($('reference-slot').value)||1;
      const body={role:state.role,sample,box:state.crop,time:state.frameTime};
      const d=await api(`/api/projects/${id}/reference`,jsonOpts(body));
      state.current.references=d.references;
      state.current.reference_samples=d.reference_samples;
      state.current.reference_counts=d.reference_counts;
      state.current.references_ready=d.references_ready;
      showReferences();
      const remaining=(d.reference_samples[state.role]||[]).findIndex(x=>!x);
      if(remaining>=0)$('reference-slot').value=String(remaining+1);
      state.crop=null;drawCrop();$('crop-hint').textContent='No selection';
      notify(`${roleNames[state.role]} photo ${sample} saved.`);
    }catch(e){notify(e.message,true)}
  }
  function drawCalibration(){const c=$('calib-canvas');const v=canvasInit(c,$('calib-frame'));if(!v)return;let {ctx,w,h}=v;ctx.clearRect(0,0,w,h);state.marked.forEach((p,i)=>{let x=p.uv[0]*w,y=p.uv[1]*h;ctx.beginPath();ctx.arc(x,y,11,0,2*Math.PI);ctx.fillStyle='#d1fe62';ctx.fill();ctx.strokeStyle='#182410';ctx.lineWidth=2;ctx.stroke();ctx.fillStyle='#192310';ctx.font='bold 11px Arial';ctx.textAlign='center';ctx.fillText(String(i+1),x,y+4);});$('calib-count').textContent=`${state.marked.length} point${state.marked.length===1?'':'s'}`;$('landmark-list').innerHTML=state.marked.map((p,i)=>`<span class="landmark-chip">${i+1}. ${escapeHtml(state.landmarks.find(l=>l.id===p.id)?.label||p.id)}</span>`).join('');}
  async function saveCalibration(){if(!state.current)return;if(state.marked.length<4){notify('Select at least four widely spaced pitch landmarks.',true);return}try{let d=await api(`/api/projects/${state.current.id}/calibrate`,jsonOpts({points:state.marked,time:state.calibTime}));state.current.calibrated=true;showReferences();notify(`Calibration saved · Median error ${d.median_error_m} m.${d.warning?' '+d.warning:''}`);}catch(e){notify(e.message,true)}}
  function startPolling(id){clearInterval(state.poll);let poll=async()=>{try{let d=await api(`/api/projects/${id}`);state.current=d.project;let p=d.project;$('processing-stage').textContent=p.stage;$('progress-fill').style.width=`${p.progress||0}%`;$('progress-text').textContent=`${p.progress||0}% · Stage progress (not frame-level)`;$('processing-log').textContent=(p.log_tail?.length?p.log_tail:['Preparing local pipeline…']).join('\n');if(p.state==='done'||p.state==='failed'){clearInterval(state.poll);await loadProjects();if(p.state==='done'){notify('Analysis complete! Your dashboard is ready.');nav('analytics')}else{notify('Processing stopped: '+(p.error||'See the processing log'),true);switchSetup('config');}}}catch(e){console.error(e)}};poll();state.poll=setInterval(poll,2600)}
  async function beginProcessing(){
    const p=state.current;
    if(!p)return;
    const count=role=>(p.reference_samples?.[role]||[]).filter(Boolean).length;
    const withBall=$('include-ball').checked;
    if(count('referee')!==3){
      notify('Please capture three referee reference photos.',true);
      return;
    }
    if(withBall && count('ball')!==3){
      notify('Ball tracking requires three ball reference photos.',true);
      return;
    }
    if(!p.calibrated){
      notify('Save pitch calibration before running the pipeline.',true);
      return;
    }
    const b=$('start-processing');b.disabled=true;
    try{
      await api(`/api/projects/${p.id}/start`,jsonOpts({with_ball:withBall}));
      switchSetup('processing');startPolling(p.id);
      notify('Analysis started. You can monitor each processing stage here.');
    }catch(e){notify(e.message,true)}finally{b.disabled=false}
  }
  function fileLink(files,label){return files.find(f=>f.label===label)}
  function videoCard(file, role, heading) {
    if (!file) return `<article class="match-video-card"><div class="video-card-heading"><strong>${heading}</strong><span>NOT GENERATED</span></div><div class="video-empty">No output yet. Check processing log for this stage.</div></article>`;
    const label = escapeHtml(file.label);
    const src = escapeHtml(file.url);
    const download = escapeHtml(file.download_url || file.url);
    const ready = Boolean(file.playable);
    return `<article class="match-video-card"><div class="video-card-heading"><strong>${escapeHtml(heading)}</strong><span class="${ready?'video-ready':'video-pending'}">${ready?'H.264 READY':'PREPARING'}</span></div>
      <div class="video-stage"><video controls preload="metadata" playsinline data-video-role="${escapeHtml(role)}" ${ready?`src="${src}"`:''} aria-label="${label}"></video><div class="video-error hidden" role="alert">Could not play this video in the browser. Try downloading the original or preparing playback again.</div></div>
      <div class="video-actions"><a href="${download}" download>↓ Download original</a>${ready?`<a href="${src}" target="_blank" rel="noopener">Open playable MP4 ↗</a>`:'<span>Preparing compatible MP4…</span>'}</div></article>`;
  }
  function videosSection(files, mediaState, mediaErrors) {
    const byRole = role => files.find(f=>f.kind==='video' && f.role===role);
    const cards = [['annotated','Annotated footage'],['birdseye',"Bird's-eye view"],['side_by_side','Side-by-side comparison']].map(([role,title])=>videoCard(byRole(role),role,title)).join('');
    const possession = byRole('possession');
    const info = mediaState==='running' ? '<span class="video-prep-note">⏳ Converting output videos into browser-compatible H.264. This can take a while for larger clips.</span>' : '';
    const warning = mediaState==='partial' ? `<span class="video-prep-note warn">${escapeHtml((mediaErrors||[]).join(' | ')||'Some videos could not be converted.')}<button id="retry-web-videos" class="small-button">Retry video conversion</button></span>` : '';
    return `<div class="analytics-section-title"><h2>Match videos</h2><span class="tiny-label">THREE VIEWING ANGLES</span></div>${info}${warning}<div class="match-video-grid">${cards}${possession?videoCard(possession,'possession','Possession replay'):''}</div>`;
  }

  function statCard(label,value,note){return `<div class="stat-card"><div class="label">${label}</div><div class="value">${value}</div><small>${note}</small></div>`}

  function imageCard(f) {
    const descriptions = {
      "Analytics overview":
        "Movement, pitch usage, tracking coverage and team shape.",
      "Team A heatmap":
        "Locations most frequently occupied by tracked Team A players.",
      "Team B heatmap":
        "Locations most frequently occupied by tracked Team B players.",
      "Team A positioning":
        "Median observed locations of Team A player tracks.",
      "Team B positioning":
        "Median observed locations of Team B player tracks.",
      "Team spread":
        "Changes in visible team width and depth."
    };

    const description =
      descriptions[f.label] || "Tracking visualization";

    return `
      <article class="visual-card insight-card">
        <div class="visual-head">
          <strong>${escapeHtml(f.label)}</strong>
        </div>
        <img
          src="${f.url}"
          alt="${escapeHtml(f.label)}"
          loading="lazy"
        />
        <div class="insight-card-footer">
          <p>${escapeHtml(description)}</p>
          <a href="${f.url}"
            target="_blank"
            rel="noopener">
            View full size ↗
          </a>
        </div>
      </article>
    `;
  }
  async function renderAnalytics(){const area=$('analytics-content');if(!state.current){area.innerHTML='<div class="empty-analytics"><h2>No match selected yet</h2><p>Upload and process a video, or open an existing match from the library.</p><button class="btn btn-primary" data-open-upload>Upload footage ↗</button></div>';return}let p=state.current;if(p.state!=='done'&&p.state!=='failed'){area.innerHTML=`<div class="empty-analytics"><h2>Results are not ready yet</h2><p>${escapeHtml(p.title)} has not finished processing.</p><button class="btn btn-primary" data-go="setup">Open match studio →</button></div>`;return}try{let d=await api(`/api/projects/${p.id}/results`),{files,summaries,players,team_shape}=d;const a=summaries.analytics||{},teams=a.team_stats||[];let tA=teams.find(t=>t.team==='Team A')||{},tB=teams.find(t=>t.team==='Team B')||{};let possession=summaries.possession||null;let videosHtml=videosSection(files,d.media_state,d.media_errors);
  let cards=[statCard('TEAM A · TRACKLET DISTANCE',fmt(tA.distance_m_est_sum_tracklets)+' m','Estimated observed movement'),statCard('TEAM B · TRACKLET DISTANCE',fmt(tB.distance_m_est_sum_tracklets)+' m','Estimated observed movement'),statCard('PLAYER TRACKLETS',fmt(a.tracklets),'IDs may split or switch'),statCard('OBSERVED COORDINATES',fmt(a.observed_rows),'Measured positions')];
  $('analytics-subtitle').textContent=`${p.title} · ${p.team_a} vs ${p.team_b} · ${fmt(p.meta.duration_s)}s video`;
  const playerHeatmaps=files.filter(f=>f.kind==='player_image');
  const shapeRows=(team_shape||[]).filter(t=>t.eligible_frames && Number(t.eligible_frames)>0);
  const shapeSection=shapeRows.length?`<div class="analytics-section-title"><h2>Team shape</h2><span class="tiny-label">VISIBLE PLAYERS ONLY</span></div><div class="stat-grid">${shapeRows.flatMap(r=>[statCard((r.team==='Team A'?p.team_a:p.team_b)+' · WIDTH',fmt(r.average_width_m)+' m', '10th–90th percentile span'),statCard((r.team==='Team A'?p.team_a:p.team_b)+' · DEPTH',fmt(r.average_depth_m)+' m','10th–90th percentile span')]).join('')}</div>`:'';
  const playerSection=playerHeatmaps.length?`<div class="analytics-section-title"><h2>Player activity maps</h2><span class="tiny-label">BY TRACK ID</span></div><div class="panel"><div class="card-topline"><strong>Select player track</strong><select id="player-heatmap-select" class="small-button">${playerHeatmaps.map((f,i)=>`<option value="${i}">${escapeHtml(f.label)}</option>`).join('')}</select></div><img id="player-heatmap-image" src="${playerHeatmaps[0].url}" alt="Selected player heatmap" style="width:100%;max-width:900px;margin:auto;display:block;aspect-ratio:1.7;object-fit:contain"></div>`:'';
  area.innerHTML=`<div class="results-top"><div class="match-overview"><div class="match-symbol">◈</div><h3>${escapeHtml(p.team_a)} <span style="color:var(--sub)">vs</span> ${escapeHtml(p.team_b)}</h3><p>Player tracking, heatmaps, distance and tactical positioning. Generated locally from your footage.</p><div class="overview-kv"><span>Duration</span><strong>${duration(p.meta.duration_s)}</strong></div><div class="overview-kv"><span>Footage</span><strong>${p.meta.width} × ${p.meta.height}</strong></div><div class="overview-kv"><span>Processing</span><strong>${p.state==='done'?'Complete':'Partial / interrupted'}</strong></div>${possession?`<div class="overview-kv"><span>Team possession · classified time</span><strong>${fmt(possession.team_only_possession_pct?.['Team A'])}% / ${fmt(possession.team_only_possession_pct?.['Team B'])}%</strong></div>`:''}</div><div class="match-replay-intro"><strong>Match videos</strong><p>Every available match view appears below — annotated player tracking, tactical bird's-eye, and comparison. These videos are processed locally and converted to H.264 for browser playback.</p><span>PLAY · SCRUB · DOWNLOAD</span></div></div>${videosHtml}<div class="stat-grid">${cards.join('')}</div>${possession?`<p class="subtle-notice">Possession is experimental. ${fmt(possession.all_time_percentage?.Unknown)}% of time is Unknown; team-only values omit unclassified frames.</p>`:''}<div class="analytics-section-title"><h2>Pitch activity and positioning</h2><span class="tiny-label">HEATMAPS / TEAM SHAPE</span></div><div class="visual-grid">${files.filter(f =>
    f.kind === 'image' &&
    [
      'Team A heatmap',
      'Team B heatmap',
      'Team A positioning',
      'Team B positioning'
    ].includes(f.label)
  ).map(imageCard).join('')||'<div class="empty-block">No charts generated yet.</div>'}</div>${fileLink(files,'Analytics overview')?`<div class="analytics-section-title"><h2>Match analysis overview</h2></div><div class="visual-grid">${imageCard(fileLink(files,'Analytics overview'))}</div>`:''}${shapeSection}${playerSection}<div class="analytics-section-title"><h2>Player movement estimates</h2><span class="tiny-label">ESTIMATES / METRES</span></div><div class="table-wrap"><table class="metrics-table"><thead><tr><th>TRACK ID</th><th>TEAM</th><th>ROLE</th><th>DISTANCE COVERED</th><th>OBSERVED FRAMES</th><th>REJECTED JUMPS</th></tr></thead><tbody>${players.slice().sort((x,y)=>Number(y.distance_m_est||0)-Number(x.distance_m_est||0)).map(row=>`<tr><td>#${escapeHtml(row.track_id)}</td><td>${row.team==='Team A'?escapeHtml(p.team_a):row.team==='Team B'?escapeHtml(p.team_b):'Unknown'}</td><td>${escapeHtml(row.role)}</td><td><strong>${fmt(row.distance_m_est)} m</strong></td><td>${fmt(row.observed_frames)}</td><td>${fmt(row.rejected_jump_segments)}</td></tr>`).join('')||'<tr><td colspan="6">No movement records yet.</td></tr>'}</tbody></table></div><div class="analytics-section-title"><h2>Data exports</h2></div><div class="downloads">${files.filter(f=>f.kind==='csv'||f.kind==='json').map(f=>`<a class="download-item" href="${f.url}" download>↓ ${escapeHtml(f.label)}</a>`).join('')}</div><p class="subtle-notice">Distance figures sum tracklet movement visible in this clip; they are not verified full-match physical totals. Track IDs are not shirt numbers. Team shape reflects players in view and is affected by calibration and limited camera visibility.</p>`;
  const playerSelect=$('player-heatmap-select');playerSelect?.addEventListener('change', e=>{const f=playerHeatmaps[Number(e.target.value)];if(f)$('player-heatmap-image').src=f.url});
  area.querySelectorAll('video[data-video-role]').forEach(video=>{
    video.addEventListener('error',()=>{
      const box=video.closest('.video-stage');
      box?.querySelector('.video-error')?.classList.remove('hidden');
    });
  });
  $('retry-web-videos')?.addEventListener('click',async()=>{
    try{await api(`/api/projects/${p.id}/prepare-videos`,jsonOpts({}));notify('Browser-video conversion started.');setTimeout(renderAnalytics,1500)}
    catch(error){notify(error.message,true)}
  });
  clearTimeout(state.videoPoll);
  if(d.media_state==='running'&&state.view==='analytics')state.videoPoll=setTimeout(renderAnalytics,2500);
  }catch(e){area.innerHTML=`<div class="empty-block">Could not load results: ${escapeHtml(e.message)}</div>`;notify(e.message,true)}}
  function bind(){document.addEventListener('click',e=>{let p=e.target.closest('[data-project]');if(p){openProject(p.dataset.project);return}if(e.target.closest('[data-open-upload]')){setupUpload();return}let b=e.target.closest('[data-go]');if(b)nav(b.dataset.go);let n=e.target.closest('[data-nav]');if(n)nav(n.dataset.nav)});
    $('theme-toggle').addEventListener('click',()=>setTheme(getTheme()==='dark'?'light':'dark'));
    $('import-existing').addEventListener('click',importExisting);
    const zone=$('upload-zone');zone.addEventListener('click',()=>$('video-file').click());zone.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();$('video-file').click()}});zone.addEventListener('dragover',e=>{e.preventDefault();zone.classList.add('dragging')});zone.addEventListener('dragleave',()=>zone.classList.remove('dragging'));zone.addEventListener('drop',e=>{e.preventDefault();zone.classList.remove('dragging');chooseFile(e.dataTransfer.files[0])});$('video-file').addEventListener('change',e=>chooseFile(e.target.files[0]));$('upload-submit').addEventListener('click',submitVideo);
    $('frame-time').addEventListener('input',e=>{clearTimeout(state.frameTimer);state.frameTimer=setTimeout(()=>setFrame(e.target.value),125)});
    $('calib-time').addEventListener('input',e=>{clearTimeout(state.calibTimer);state.calibTimer=setTimeout(()=>setCalibFrame(e.target.value),125)});
    $('calib-time-input').addEventListener('change',e=>setCalibFrame(e.target.value));
    $('calib-prev').addEventListener('click',()=>setCalibFrame(state.calibTime-1/(state.current?.meta.fps||30)));
    $('calib-next').addEventListener('click',()=>setCalibFrame(state.calibTime+1/(state.current?.meta.fps||30)));$('config-frame').addEventListener('load',drawCrop);$('calib-frame').addEventListener('load',drawCalibration);window.addEventListener('resize',()=>{drawCrop();drawCalibration()});attachCrop();
    document.querySelectorAll('[data-role]').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('[data-role]').forEach(x=>x.classList.remove('selected'));b.classList.add('selected');state.role=b.dataset.role;
      $('reference-slot').innerHTML=Array.from({length:3},(_,i)=>`<option value="${i+1}">Photo ${i+1} / 3</option>`).join('');
      const samples=state.current?.reference_samples?.[state.role]||[];
      $('reference-slot').value=String(Math.max(1,samples.findIndex(x=>!x)+1));
      state.crop=null;drawCrop();showReferences();}));
    $('reference-preview').addEventListener('click',e=>{const b=e.target.closest('[data-ref-slot]');if(b)$('reference-slot').value=b.dataset.refSlot});
    $('save-reference').addEventListener('click',saveReference);
    $('calib-canvas').addEventListener('click',e=>{const select=$('landmark-select');let id=select.value;if(state.marked.some(p=>p.id===id)){notify('This landmark is already selected. Choose a different one or undo.',true);return}state.marked.push({id,uv:canvasUV(e,$('calib-canvas'))});drawCalibration();});$('undo-landmark').addEventListener('click',()=>{state.marked.pop();drawCalibration()});$('save-calibration').addEventListener('click',saveCalibration);$('start-processing').addEventListener('click',beginProcessing);
  }
  async function main(){setTheme(getTheme());bind();try{const d=await api('/api/landmarks');state.landmarks=d.landmarks;$('landmark-select').innerHTML=d.landmarks.map(l=>`<option value="${escapeHtml(l.id)}">${escapeHtml(l.label)}</option>`).join('')}catch(e){notify(e.message,true)}await loadProjects();await loadLocalMatches();}
  main();
})();
