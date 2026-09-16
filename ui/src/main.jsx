import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import HRChatUI from './HRChatUI.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <HRChatUI />
  </StrictMode>,
)
