import express from "express";
import dotenv from "dotenv";
import twilio from "twilio";
import { GoogleGenAI } from "@google/genai";
import wav from "wav";
import fs from "fs";
import path from "path";
import crypto from "crypto";
import { fileURLToPath } from "url";

dotenv.config();

const {
  PORT = 3000,
  BASE_URL,
  GEMINI_API_KEY,
  TWILIO_AUTH_TOKEN,
  NODE_ENV = "development",
} = process.env;

if (!BASE_URL || !GEMINI_API_KEY || !TWILIO_AUTH_TOKEN) {
  throw new Error(
    "Missing BASE_URL, GEMINI_API_KEY, or TWILIO_AUTH_TOKEN in your .env file."
  );
}

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const ai = new GoogleGenAI({ apiKey: GEMINI_API_KEY });

app.use(express.urlencoded({ extended: true }));
app.use(express.json());

const audioDir = path.join(__dirname, "public", "audio");
fs.mkdirSync(audioDir, { recursive: true });
app.use("/audio", express.static(audioDir));

const sessions = new Map();

const SYSTEM_PROMPT = `
You are CareRing, a warm, patient, elderly-friendly phone companion.

Your role on the phone:
- Speak simply, slowly, and clearly.
- Keep replies short: 1–2 sentences.
- Ask only one question at a time.
- Be calm and supportive.
- Avoid jargon or complicated words.

Conversation style rules:
- Maximum 30 words
- One gentle question
- Sound human and warm
`;

function getSession(callSid) {
  if (!sessions.has(callSid)) {
    sessions.set(callSid, {
      history: [],
      createdAt: new Date().toISOString(),
    });
  }

  return sessions.get(callSid);
}

async function generateReply(callSid, userText) {
  const session = getSession(callSid);

  session.history.push({ role: "user", text: userText });

  const conversation = session.history
    .slice(-10)
    .map((m) => `${m.role === "user" ? "Caller" : "CareRing"}: ${m.text}`)
    .join("\n");

  const prompt = `${SYSTEM_PROMPT}

Conversation:
${conversation}

Reply as CareRing.`;

  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash",
    contents: [{ parts: [{ text: prompt }] }],
  });

  const text = response.text?.trim() || "I'm here with you. How are you feeling right now?";

  session.history.push({ role: "assistant", text });

  return text;
}

async function saveWaveFile(filename, pcmData, channels = 1, rate = 24000, sampleWidth = 2) {
  return new Promise((resolve, reject) => {
    const writer = new wav.FileWriter(filename, {
      channels,
      sampleRate: rate,
      bitDepth: sampleWidth * 8,
    });

    writer.on("finish", resolve);
    writer.on("error", reject);
    writer.write(pcmData);
    writer.end();
  });
}

async function synthesizeSpeech(text) {
  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash-preview-tts",
    contents: [{ parts: [{ text }] }],
    config: {
      responseModalities: ["AUDIO"],
      speechConfig: {
        voiceConfig: {
          prebuiltVoiceConfig: { voiceName: "Kore" },
        },
      },
    },
  });

  const data = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;

  if (!data) {
    throw new Error("No audio returned from Gemini TTS");
  }

  const audioBuffer = Buffer.from(data, "base64");
  const filename = `${crypto.randomUUID()}.wav`;
  const filepath = path.join(audioDir, filename);

  await saveWaveFile(filepath, audioBuffer);

  return `${BASE_URL}/audio/${filename}`;
}

function isRequestFromTwilio(req) {
  if (NODE_ENV !== "production") return true;

  const signature = req.header("X-Twilio-Signature");
  if (!signature) return false;

  const url = `${BASE_URL}${req.originalUrl}`;

  return twilio.validateRequest(TWILIO_AUTH_TOKEN, signature, url, req.body);
}

function buildGather(twiml) {
  const gather = twiml.gather({
    input: "speech",
    action: `${BASE_URL}/gather`,
    method: "POST",
    speechTimeout: "auto",
    language: "en-US",
    actionOnEmptyResult: true,
  });

  gather.pause({ length: 1 });
}

app.get("/health", (_req, res) => {
  res.json({ ok: true });
});

app.post("/voice", async (req, res) => {
  console.log("VOICE HIT", req.body);

  if (!isRequestFromTwilio(req)) {
    return res.status(403).send("Invalid Twilio signature");
  }

  const twiml = new twilio.twiml.VoiceResponse();
  const callSid = req.body.CallSid;

  getSession(callSid);

  try {
    const greeting = "Hello, this is CareRing checking in. Take your time. How are you feeling today?";

    const audioUrl = await synthesizeSpeech(greeting);

    twiml.play(audioUrl);

    buildGather(twiml);

    res.type("text/xml").send(twiml.toString());
  } catch (error) {
    console.error(error);

    twiml.say("Hello, this is CareRing checking in. How are you feeling today?");

    buildGather(twiml);

    res.type("text/xml").send(twiml.toString());
  }
});

app.post("/gather", async (req, res) => {
  console.log("GATHER HIT", req.body);

  if (!isRequestFromTwilio(req)) {
    return res.status(403).send("Invalid Twilio signature");
  }

  const twiml = new twilio.twiml.VoiceResponse();
  const callSid = req.body.CallSid;
  const transcript = (req.body.SpeechResult || "").trim();

  try {
    let reply;

    if (!transcript) {
      reply = "That's okay. Take your time. Could you please say that again?";
    } else {
      reply = await generateReply(callSid, transcript);
    }

    const audioUrl = await synthesizeSpeech(reply);

    twiml.play(audioUrl);

    buildGather(twiml);

    res.type("text/xml").send(twiml.toString());
  } catch (error) {
    console.error(error);

    twiml.say("I'm sorry, I had trouble hearing that. Could you say it again?");

    buildGather(twiml);

    res.type("text/xml").send(twiml.toString());
  }
});

app.listen(PORT, () => {
  console.log(`CareRing voicebot running on port ${PORT}`);
  console.log(`Base URL: ${BASE_URL}`);
});