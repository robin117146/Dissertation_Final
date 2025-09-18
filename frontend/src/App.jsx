import React, { useEffect, useState } from 'react'

export default function App(){
  const [pred, setPred] = useState(null)
  useEffect(()=>{
    fetch('/api/predict', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({lag1:1200, lag2:1180, weekofyear:40})
    }).then(r=>r.json()).then(setPred).catch(e=>console.log(e))
  },[])
  return (
    <div style={{fontFamily:'Arial, sans-serif', padding:20}}>
      <h1>FluPredict Prototype</h1>
      <p>Simple demo frontend that calls the backend predict endpoint.</p>
      <pre>{pred ? JSON.stringify(pred, null, 2) : 'Loading prediction (or models not present)...'}</pre>
    </div>
  )
}
