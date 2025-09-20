const axios = require('axios');

const sendLocation = async (touristId, lat, lon) => {
  try {
    const res = await axios.post('http://flask-server:8080/ingest/ping', {
      tourist_id: touristId,
      lat: lat,
      lon: lon,
      ts: Math.floor(Date.now()/1000)
    });
    console.log(res.data);
  } catch (err) {
    console.error(err);
  }
};


const io = require('socket.io-client');
const socket = io('http://flask-server:8080');

socket.on('connect', () => {
  console.log('Connected to Flask SocketIO');
  socket.emit('subscribe', { tourist_id: 123 });
});

socket.on('geofence_alert', (alert) => {
  console.log('ALERT:', alert);
});

