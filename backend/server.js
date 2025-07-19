const express = require('express');
const cors = require('cors');
const path = require('path');
const app = express();
const port = 3001;

app.use(cors());
app.use(express.json());

// Serve static files from the ai-interview-proctoring directory under the right route
const staticPath = path.join(__dirname, '../ai-interview-proctoring');
console.log('Serving static files from:', staticPath);
app.use('/ai-interview-proctoring', express.static(staticPath));

// Serve the main page at root
app.get('/', (req, res) => {
  const indexPath = path.join(staticPath, 'index.html');
  console.log('Serving index from:', indexPath);
  res.sendFile(indexPath);
});

// Add a test endpoint
app.get('/test', (req, res) => {
  res.json({ message: 'Server is running', staticPath: staticPath });
});

const userIds = new Set();

app.post('/api/store-userid', (req, res) => {
  const { userId } = req.body;
  if (!userId) {
    return res.status(400).json({ error: 'userId is required' });
  }
  userIds.add(userId);
  console.log('Stored userId:', userId);
  res.json({ message: 'User ID stored successfully' });
});

app.listen(port, () => {
  console.log(`🚀 Backend server listening at http://localhost:${port}`);
});
