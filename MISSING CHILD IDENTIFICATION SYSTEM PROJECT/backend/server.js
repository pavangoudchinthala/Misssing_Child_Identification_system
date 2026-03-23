const express = require('express');
const multer = require('multer');
const path = require('path');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT || 3000;

/* ---------------- MIDDLEWARE ---------------- */
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Serve static files
app.use(express.static(path.join(__dirname, 'public')));
app.use('/images', express.static(path.join(__dirname, 'public/images')));

/* ---------------- TEMP STORAGE (REPLACES MYSQL) ---------------- */
let children = [];

/* ---------------- MULTER SETUP ---------------- */
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, path.join(__dirname, 'public/images'));
  },
  filename: function (req, file, cb) {
    const uniqueName = Date.now() + "-" + file.originalname.replace(/\s+/g, '_');
    cb(null, uniqueName);
  }
});

const upload = multer({ storage: storage });

/* ---------------- HOME PAGE ---------------- */
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

/* ---------------- ADD MISSING CHILD ---------------- */
app.post('/addMissingChild', upload.single('photo'), (req, res) => {
  const child = req.body;

  if (req.file) {
    child.photo = req.file.filename;
  }

  children.push(child);

  console.log("DATA:", child);
  res.send('Missing child added successfully');
});

/* ---------------- FIND MISSING CHILD ---------------- */
app.get('/findMissingChild/:name', (req, res) => {
  const name = req.params.name;

  const child = children.find(c => c.name === name);

  if (child) {
    res.json(child);
  } else {
    res.json(null);
  }
});

/* ---------------- DISPLAY CHILD PAGE ---------------- */
app.get('/displayChild', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'child.html'));
});

/* ---------------- START SERVER ---------------- */
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});