import logo from "./logo.svg";
import "./App.css";
import * as tf from "@tensorflow/tfjs";

import React from "react";
import SpeechRecognition, {
  useSpeechRecognition,
} from "react-speech-recognition";
import {
  Hearing,
  MicOffRounded,
  MicRounded,
  PlayCircleFilledOutlined,
  SentimentSatisfied,
  SentimentVerySatisfied,
  VolumeDown,
  VolumeUp,
} from "@material-ui/icons";
import { IconButton, Typography } from "@material-ui/core";
import { useEffect, useState } from "react";

import { Slider, Grid } from "@material-ui/core";

// import Speech from "react-speech";

const speakUp = (text, vol, pitch, lang, end) => {
  var u = new SpeechSynthesisUtterance();
  u.text = text;
  u.volume = vol;
  u.rate = pitch;
  u.lang = lang;
  speechSynthesis.speak(u);
  u.onend = end;
};

function App() {
  const [nlpModel, setNlpModel] = useState(null);
  const [wordIndex, setWordIndex] = useState({});
  const [wordToken, setWordToken] = useState({});
  const [phrase, setPhrase] = useState("Hello there its me Neo");
  const [genLen, setGenLen] = useState(10);
  const [speaking, setSpeaking] = useState(false);
  const [volume, setVolume] = useState(0.8);
  const [pitch, setPitch] = useState(0.8);
  const {
    transcript,
    listening,
    resetTranscript,
    browserSupportsSpeechRecognition,
  } = useSpeechRecognition();

  async function loadModel() {
    const model = await tf.loadLayersModel("/static/model.json");
    console.log("model loaded");
    setNlpModel(model);
  }

  useEffect(() => {
    loadModel();
    fetch("/static/word_index.json")
      .then((response) => response.json())
      .then((data) => setWordIndex(data));
    fetch("/static/word_token.json")
      .then((response) => response.json())
      .then((data) => setWordToken(data));
  }, []);

  const genText = (text) => {
    var phrase = "";
    if (nlpModel !== null) {
      for (let i = 0; i <= genLen; i++) {
        var token = text.toLowerCase().split(" ");
        var tok = token.map((t) =>
          wordIndex[t.toLowerCase()] !== undefined
            ? wordIndex[t.toLowerCase()]
            : 0
        );
        var inputStr = Array(wordToken["max_seq_len"] - 1 - tok.length)
          .fill(0)
          .concat(tok);
        const prediction = nlpModel
          .predict(tf.tensor([inputStr]))
          .argMax(-1)
          .dataSync();
        var txt = wordToken[prediction[0]];
        if (txt[txt.length - 1] === "0") {
          text += ` ${txt.replace("0", ".")}`;
          phrase += ` ${txt.replace("0", ".")}`;
          break;
        }
        text += ` ${txt}`;
        phrase += ` ${txt}`;
      }
      return phrase;
    }
    return "";
  };

  useEffect(() => {
    if (!listening && transcript !== "") {
      setSpeaking("true");
      console.log(speaking);
      var toSpeak = genText(transcript);
      setPhrase(toSpeak);
      speakUp(toSpeak, volume, pitch, "en-US", () => setSpeaking(false));
    }
  }, [transcript, listening]);

  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} style={{ height: "8vh" }}></img>
      </header>
      {!browserSupportsSpeechRecognition ? (
        <span>Browser doesn't support speech recognition.</span>
      ) : (
        ""
      )}
      <Grid container spacing={2} style={{ width: "50vh", margin: "auto" }}>
        <Grid item>
          <VolumeDown />
        </Grid>
        <Grid item xs>
          <Slider
            value={volume}
            onChange={(e, v) => setVolume(v)}
            aria-labelledby="continuous-slider"
            step={0.001}
            max={1}
            min={0.001}
          />
        </Grid>
        <Grid item>
          <VolumeUp />
        </Grid>
      </Grid>

      <Grid container spacing={2} style={{ width: "50vh", margin: "auto" }}>
        <Grid item>
          <SentimentSatisfied />
        </Grid>
        <Grid item xs>
          <Slider
            value={pitch}
            onChange={(e, v) => setPitch(v)}
            aria-labelledby="continuous-slider"
            step={0.001}
            max={1}
            min={0.001}
          />
        </Grid>
        <Grid item>
          <SentimentVerySatisfied />
        </Grid>
      </Grid>

      <IconButton
        size="medium"
        onClick={
          listening
            ? SpeechRecognition.stopListening
            : SpeechRecognition.startListening
        }
      >
        {listening ? (
          <MicRounded fontSize="large" color="secondary"></MicRounded>
        ) : speaking ? (
          <PlayCircleFilledOutlined color="secondary" fontSize="large" />
        ) : (
          <MicOffRounded fontSize="large" color="primary"></MicOffRounded>
        )}
      </IconButton>
      <Typography color="primary">{transcript ? transcript : "-"}</Typography>
      <Typography color="secondary">{phrase ? phrase : "-"}</Typography>
    </div>
  );
}

export default App;
