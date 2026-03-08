// CareRing backend — Express + Twilio + Gemini
// Inbound call flow: elder calls the Twilio number → Gemini answers
// Outbound call flow: app hits POST /call-elder → Twilio dials elder → Gemini answers
// Usage: node src/server.js
// Requires: npm install express twilio @google/generative-ai dotenv

import dotenv from 'dotenv';
dotenv.config();
import express from 'express';
import twilio from 'twilio';
import { GoogleGenerativeAI } from '@google/generative-ai';

const app = express();
app.use(express.json());
app.use(express.urlencoded({ extended: false }));

// ── CORS ──
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Headers', 'Content-Type');
  res.header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  if (req.method === 'OPTIONS') return res.sendStatus(200);
  next();
});

// ── CONFIG ──
const BASE_URL        = process.env.BASE_URL;
const GEMINI_KEY      = process.env.GEMINI_API_KEY;
const TWILIO_FROM_NUM = '+441923311452';
const PORT            = process.env.PORT || 3000;
const twilioClient = twilio('ACa1e08eaaee5c2aed6266c4a899f8c898', 'deda1f05d19d8e354c2c237f1c2d1a93');
const genAI        = new GoogleGenerativeAI(GEMINI_KEY);

// ── IN-MEMORY SESSION STORE ──
const sessions = new Map();

// ══════════════════════════════════════════════
//  POST /call-elder  — app calls this to trigger an outbound call
//  Body: { phoneNumber: "+15550001234", elderName: "Margaret" }
// ══════════════════════════════════════════════
app.post('/call-elder', async (req, res) => {
  const { phoneNumber, elderName } = req.body;

  if (!phoneNumber) {
    return res.status(400).json({ ok: false, error: 'phoneNumber is required' });
  }

console.log(`[call-elder] dialling ${phoneNumber} for ${elderName || 'elder'}`);
console.log(`[call-elder] from: ${TWILIO_FROM_NUM}`);
console.log(`[call-elder] SID: ACa1e08eaaee5c2aed6266c4a899f8c898`);

  try {
    const call = await twilioClient.calls.create({
      to:   phoneNumber,
      from: TWILIO_FROM_NUM,
      // When elder picks up, Twilio hits /voice-outbound which seeds the session
      // with their name before handing off to the normal AI flow
      url:  `${BASE_URL}/voice-outbound?elderName=${encodeURIComponent(elderName || 'there')}`,
      method: 'POST',
      statusCallback: `${BASE_URL}/call-status`,
      statusCallbackMethod: 'POST',
    });

    console.log(`[call-elder] call created: ${call.sid}`);
    res.json({ ok: true, callSid: call.sid });

  } catch (err) {
    console.error('[call-elder] Twilio error:', err.message);
    res.status(500).json({ ok: false, error: err.message });
  }
});

// ══════════════════════════════════════════════
//  POST /voice-outbound  — webhook when outbound call is answered
//  Same as /voice but elderName comes from query param
// ══════════════════════════════════════════════
app.post('/voice-outbound', async (req, res) => {
  const callSid   = req.body.CallSid;
  const elderName = req.query.elderName || 'there';

  console.log(`[voice-outbound] ${callSid} answered — elder: ${elderName}`);

  sessions.set(callSid, {
    history:   [],
    topics:    ['medication', 'day', 'activity', 'help', 'sleep'],
    elderName,
    createdAt: new Date().toISOString(),
  });

  const greeting = await getAiResponse(callSid, null);

  const twiml  = new twilio.twiml.VoiceResponse();
  const gather = twiml.gather({
    input: 'speech', action: `${BASE_URL}/respond`,
    method: 'POST', speechTimeout: 'auto', language: 'en-US',
  });
  gather.say({ voice: 'Polly.Joanna-Neural' }, greeting);

  twiml.say({ voice: 'Polly.Joanna-Neural' }, "I didn't catch that — take your time and speak when you're ready.");
  twiml.redirect({ method: 'POST' }, `${BASE_URL}/voice-outbound?elderName=${encodeURIComponent(elderName)}`);

  res.type('text/xml').send(twiml.toString());
});

// ══════════════════════════════════════════════
//  POST /call-status  — optional: Twilio calls this with call lifecycle events
// ══════════════════════════════════════════════
app.post('/call-status', (req, res) => {
  const { CallSid, CallStatus } = req.body;
  console.log(`[call-status] ${CallSid} → ${CallStatus}`);
  res.sendStatus(200);
});

// ══════════════════════════════════════════════
//  POST /voice  — Twilio webhook when elder's INBOUND call connects
//  Set this as "A call comes in" in your Twilio phone number config
// ══════════════════════════════════════════════
app.post('/voice', async (req, res) => {
  const callSid   = req.body.CallSid;
  const callerNum = req.body.From || 'unknown';

  console.log(`[voice] inbound call ${callSid} from ${callerNum}`);

  sessions.set(callSid, {
    history:   [],
    topics:    ['medication', 'day', 'activity', 'help', 'sleep'],
    elderName: 'there',
    createdAt: new Date().toISOString(),
  });

  const greeting = await getAiResponse(callSid, null);

  const twiml  = new twilio.twiml.VoiceResponse();
  const gather = twiml.gather({
    input: 'speech', action: `${BASE_URL}/respond`,
    method: 'POST', speechTimeout: 'auto', language: 'en-US',
  });
  gather.say({ voice: 'Polly.Joanna-Neural' }, greeting);

  twiml.say({ voice: 'Polly.Joanna-Neural' }, "I didn't catch that — take your time and speak when you're ready.");
  twiml.redirect({ method: 'POST' }, `${BASE_URL}/voice`);

  res.type('text/xml').send(twiml.toString());
});

// ══════════════════════════════════════════════
//  POST /respond  — called after each elder utterance
// ══════════════════════════════════════════════
app.post('/respond', async (req, res) => {
  const callSid    = req.body.CallSid;
  const speechText = req.body.SpeechResult || '';
  const confidence = parseFloat(req.body.Confidence || '0');

  console.log(`[respond] ${callSid} — "${speechText}" (conf ${confidence.toFixed(2)})`);

  const session = sessions.get(callSid);
  if (!session) {
    const twiml = new twilio.twiml.VoiceResponse();
    twiml.say({ voice: 'Polly.Joanna-Neural' }, "I'm sorry, I lost track of our conversation. Goodbye!");
    twiml.hangup();
    return res.type('text/xml').send(twiml.toString());
  }

  if (!speechText || (confidence < 0.4 && speechText.length < 3)) {
    const twiml  = new twilio.twiml.VoiceResponse();
    const gather = twiml.gather({
      input: 'speech', action: `${BASE_URL}/respond`,
      method: 'POST', speechTimeout: 'auto', language: 'en-US',
    });
    gather.say({ voice: 'Polly.Joanna-Neural' }, "Sorry, I didn't quite catch that — could you say that again?");
    return res.type('text/xml').send(twiml.toString());
  }

  const aiReply   = await getAiResponse(callSid, speechText);
  const twiml     = new twilio.twiml.VoiceResponse();
  const isGoodbye = /goodbye|take care|have a (great|wonderful|nice) (day|rest)|talk (to you |with you )?soon/i.test(aiReply);

  if (isGoodbye) {
    twiml.say({ voice: 'Polly.Joanna-Neural' }, aiReply);
    twiml.pause({ length: 1 });
    twiml.hangup();
    sessions.delete(callSid);
    console.log(`[respond] call ${callSid} ended`);
  } else {
    const gather = twiml.gather({
      input: 'speech', action: `${BASE_URL}/respond`,
      method: 'POST', speechTimeout: 'auto', language: 'en-US',
    });
    gather.say({ voice: 'Polly.Joanna-Neural' }, aiReply);
    twiml.say({ voice: 'Polly.Joanna-Neural' }, "Are you still there? Take your time.");
    twiml.redirect({ method: 'POST' }, `${BASE_URL}/respond`);
  }

  res.type('text/xml').send(twiml.toString());
});

// ══════════════════════════════════════════════
//  Gemini helper
// ══════════════════════════════════════════════
const TOPIC_PROMPTS = {
  medication: 'Ask whether they have taken their medication today.',
  day:        'Ask how their day is going and how they are feeling emotionally.',
  activity:   'Ask what activities or things they did today.',
  help:       'Ask if they need any help or have any concerns.',
  sleep:      'Ask how well they slept last night.',
};

async function getAiResponse(callSid, userMessage) {
  const session = sessions.get(callSid);
  if (!session) return "I'm sorry, something went wrong. Goodbye!";

  if (userMessage !== null) {
    session.history.push({ role: 'user', parts: [{ text: userMessage }] });
  }

  const topicList = session.topics
    .map((id, i) => `${i + 1}. ${TOPIC_PROMPTS[id] || id}`)
    .join('\n');

  // Use elder's actual name if we have it
  const nameGreeting = session.elderName && session.elderName !== 'there'
    ? session.elderName
    : 'there';

  const systemInstruction =
`You are CareRing, a warm and caring AI assistant on a check-in phone call with an elderly person named ${nameGreeting}.

Your job is to cover these topics one at a time, in a natural conversational way:
${topicList}

Rules:
- Be warm, patient, and speak plainly — short simple sentences.
- Address them by name (${nameGreeting}) occasionally to keep it personal.
- Ask ONE topic at a time. After they answer, move to the next.
- When all topics are covered, thank them warmly and say goodbye.
- Keep every response SHORT — 1 to 3 sentences. This is a phone call.
- Never give medical advice. If something sounds urgent, tell them their caretaker will be notified.
- Do NOT say "As an AI" or anything robotic. Sound like a kind, caring human.
- On the very first turn, greet them warmly by name and jump straight into the first topic.`;

  try {
    const model      = genAI.getGenerativeModel({ model: 'gemini-2.0-flash', systemInstruction });
    const chatHistory = userMessage !== null ? session.history.slice(0, -1) : [];
    const chat        = model.startChat({ history: chatHistory });
    const prompt      = userMessage === null
      ? 'Hello, the call just connected. Please greet the person warmly and begin the first check-in topic.'
      : userMessage;

    const result = await chat.sendMessage(prompt);
    const aiText = result.response.text();

    session.history.push({ role: 'model', parts: [{ text: aiText }] });
    console.log(`[gemini] ${callSid}: "${aiText.substring(0, 100)}"`);
    return aiText;

  } catch (err) {
    console.error('[gemini] error:', err.message);
    return "I'm having a little trouble — bear with me for just a moment.";
  }
}

// ══════════════════════════════════════════════
//  GET /health
// ══════════════════════════════════════════════
app.get('/health', (req, res) => {
  res.json({ ok: true, sessions: sessions.size, time: new Date().toISOString(), baseUrl: BASE_URL });
});

app.listen(PORT, () => {
  console.log(`\n🚀 CareRing server running on port ${PORT}`);
  console.log(`   BASE_URL : ${BASE_URL}`);
  console.log(`   Health   : ${BASE_URL}/health`);
  console.log(`\n   👉 Twilio inbound webhook → ${BASE_URL}/voice  (HTTP POST)`);
  console.log(`   👉 Outbound call trigger  → POST ${BASE_URL}/call-elder\n`);
});