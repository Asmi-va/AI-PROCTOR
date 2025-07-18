const express = require('express');
const cors = require('cors');
const app = express();
const port = 3001;

app.use(cors());
app.use(express.json());

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
  console.log(`Backend server listening at http://localhost:${port}`);
});
