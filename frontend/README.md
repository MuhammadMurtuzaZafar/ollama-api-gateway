# Ollama API Gateway - Frontend# Ollama API Gateway - Frontend



Modern React-based dashboard for managing Ollama API Gateway with real-time statistics, API key management, and interactive playground.React-based frontend for the Ollama API Gateway with admin and user dashboards.



## 🚀 Features## Features



- **Admin Dashboard** - Comprehensive API key management and system-wide analytics- 🔐 **API Key Authentication** - Secure sign-in with API keys

- **User Dashboard** - Personal usage statistics and metrics- 👥 **Role-Based Access** - Admin and user dashboards with different permissions

- **Interactive Playground** - Test AI models with customizable parameters- 📊 **Analytics Dashboard** - Real-time usage statistics and charts

- **Dark Mode** - Full dark mode support with theme persistence- 🎨 **Dark/Light Theme** - Toggle between dark and light modes

- **Responsive Design** - Mobile-first design with Tailwind CSS- 📈 **Data Visualization** - Chart.js powered charts for usage metrics

- **Real-time Updates** - Live statistics and usage tracking- 🔑 **API Key Management** - Create, view, and revoke API keys (admin only)

- **API Documentation** - Built-in developer documentation with code examples

## Tech Stack

## 📋 Prerequisites

- **React 18** - UI framework

- Node.js 18+ and npm/yarn- **Vite** - Build tool

- Docker and Docker Compose (for containerized deployment)- **React Router** - Client-side routing

- Backend API running (see backend README)- **Tailwind CSS** - Styling

- **Chart.js** - Data visualization

## 🛠️ Technology Stack- **Axios** - HTTP client

- **Lucide React** - Icons

- **Framework**: React 18.3.1- **date-fns** - Date formatting

- **Build Tool**: Vite 5.4.11

- **Routing**: React Router DOM 6.28.0## Setup

- **HTTP Client**: Axios 1.7.9

- **Styling**: Tailwind CSS 3.4.17### Installation

- **Icons**: Lucide React 0.468.0

- **Charts**: Recharts 2.15.0```bash

cd frontend

## 📁 Project Structurenpm install

```

```

frontend/### Environment Variables

├── public/

│   └── assets/Create a `.env` file in the frontend directory:

│       ├── TPS-Logo.png           # Company logo

│       ├── logo.svg               # App logo (main)```env

│       ├── logo-icon.svg          # App iconVITE_API_URL=http://localhost:8000

│       ├── logo-horizontal.svg    # Horizontal logo```

│       └── favicon.svg            # Browser favicon

├── src/### Development

│   ├── components/

│   │   ├── APIKeyForm.jsx         # Create API key form```bash

│   │   ├── Charts.jsx             # Dashboard chartsnpm run dev

│   │   ├── Navbar.jsx             # Navigation bar```

│   │   └── StatsCard.jsx          # Statistics card component

│   ├── context/The frontend will be available at `http://localhost:3000`

│   │   ├── AuthContext.jsx        # Authentication state

│   │   └── ThemeContext.jsx       # Theme management### Build for Production

│   ├── pages/

│   │   ├── AdminDashboard.jsx     # Admin panel```bash

│   │   ├── Home.jsx               # Landing page with API docsnpm run build

│   │   ├── Playground.jsx         # Interactive model testing```

│   │   ├── SignIn.jsx             # Authentication page

│   │   └── UserDashboard.jsx      # User statisticsThe production build will be in the `dist` directory.

│   ├── services/

│   │   └── api.js                 # API client service### Preview Production Build

│   ├── App.jsx                    # Main app component

│   ├── index.css                  # Global styles```bash

│   └── main.jsx                   # Entry pointnpm run preview

├── Dockerfile                     # Docker configuration```

├── package.json                   # Dependencies

├── postcss.config.js              # PostCSS config## Project Structure

├── tailwind.config.js             # Tailwind configuration

├── vite.config.js                 # Vite configuration```

└── README.md                      # This filefrontend/

```├── src/

│   ├── components/       # Reusable UI components

## ⚙️ Environment Variables│   │   ├── Navbar.jsx

│   │   ├── StatsCard.jsx

Create a `.env` file in the frontend directory:│   │   ├── Charts.jsx

│   │   └── APIKeyForm.jsx

```env│   ├── context/          # React context providers

# Backend API URL│   │   ├── AuthContext.jsx

VITE_API_URL=http://localhost:8000│   │   └── ThemeContext.jsx

```│   ├── pages/            # Page components

│   │   ├── Home.jsx

**Note**: In Docker, this is automatically configured to `http://localhost:8000`│   │   ├── SignIn.jsx

│   │   ├── UserDashboard.jsx

## 🐳 Docker Deployment (Recommended)│   │   └── AdminDashboard.jsx

│   ├── services/         # API service layer

### Using Docker Compose (from project root)│   │   └── api.js

│   ├── App.jsx           # Main app component

1. **Start all services**:│   ├── main.jsx          # Entry point

```bash│   └── index.css         # Global styles

docker compose up -d├── public/               # Static assets

```├── index.html

├── package.json

2. **Access the application**:├── vite.config.js

   - Frontend: http://localhost:3000├── tailwind.config.js

   - Backend API: http://localhost:8000└── postcss.config.js

```

3. **View logs**:

```bash## Usage

docker compose logs -f frontend

```### Demo API Keys



4. **Stop services**:For testing, you can use these demo API keys:

```bash

docker compose down- **Admin**: `demo-admin-key-12345`

```- **User**: `demo-user-key-67890`



5. **Rebuild after changes**:### Admin Dashboard

```bash

docker compose up --build -dAdmins can:

```- View all API keys

- Create new API keys

### Docker Configuration- Revoke existing API keys

- View system-wide statistics

The frontend runs in a Node.js Alpine container:- Monitor usage across all users

- **Platform**: linux/amd64 (for Apple Silicon compatibility)

- **Port**: 3000 (mapped to host)### User Dashboard

- **Hot Reload**: Enabled with volume mounts

- **Dependencies**: Cached in anonymous volumeUsers can:

- View their own usage statistics

## 🔧 Local Development (Without Docker)- See requests over time

- Monitor rate limit usage

### 1. Install Dependencies- View response time metrics

- Analyze requests by endpoint

```bash

cd frontend## API Integration

npm install

```The frontend communicates with the FastAPI backend at `http://localhost:8000`. All API calls include the `X-API-Key` header for authentication.



### 2. Configure Environment### Available Endpoints



```bash- `GET /api/keys` - List API keys (requires authentication)

cp .env.example .env- `POST /api/keys` - Create new API key (admin only)

# Edit VITE_API_URL if backend is not on localhost:8000- `DELETE /api/keys/:id` - Revoke API key (admin only)

```- `GET /api/stats` - Get basic statistics

- `GET /api/stats/detailed` - Get detailed analytics

### 3. Start Development Server

## Customization

```bash

npm run dev### Changing Theme Colors

```

Edit `tailwind.config.js` to customize the primary color:

The application will be available at:

- **Frontend**: http://localhost:3000```javascript

- **Vite HMR**: Enabled for instant updatestheme: {

  extend: {

### 4. Build for Production    colors: {

      primary: {

```bash        // Customize these values

npm run build        500: '#0ea5e9',

```        600: '#0284c7',

        700: '#0369a1',

Built files will be in the `dist/` directory.      }

    }

### 5. Preview Production Build  }

}

```bash```

npm run preview

```### API Base URL



## 🎨 Available ScriptsChange the API base URL in `src/services/api.js` or use the `VITE_API_URL` environment variable.



| Command | Description |## License

|---------|-------------|

| `npm run dev` | Start development server with HMR |MIT

| `npm run build` | Build for production |
| `npm run preview` | Preview production build |
| `npm run lint` | Run ESLint (if configured) |

## 🔑 Authentication

### Demo Credentials

For testing purposes, use these API keys:

- **Admin Access**:
  - API Key: `demo-admin-key-12345`
  - Features: Full access to all dashboards, API key management

- **User Access**:
  - API Key: `demo-user-key-67890`
  - Features: Personal dashboard, usage statistics

### Sign In Flow

1. Navigate to http://localhost:3000
2. Click "Sign In with API Key"
3. Enter one of the demo keys above
4. You'll be redirected to the appropriate dashboard

## 📱 Features Guide

### Home Page
- API documentation for developers
- Code examples (cURL, Python, JavaScript)
- Endpoint reference
- Authentication guide
- Copy-to-clipboard for all code snippets

### Admin Dashboard
- **API Keys Management**:
  - Create new keys with custom names and roles
  - View all keys with preview (masked)
  - Delete keys
  - Real-time key count
  
- **System Statistics**:
  - Total requests across all keys
  - Total tokens consumed
  - Average response time
  - Active users count

- **Usage Charts**:
  - Request trends over time
  - Token usage distribution
  - Response time metrics

### User Dashboard
- **Personal Statistics**:
  - Your total requests
  - Your token usage
  - Your average response time
  - Last request timestamp

- **Usage Charts**:
  - Personal request trends
  - Token consumption over time
  - Performance metrics

### Playground
- **Model Settings** (Left Sidebar):
  - Model selection (dropdown with available models)
  - Temperature slider (0.0 - 2.0)
  - Top P slider (0.0 - 1.0)
  - Max Tokens input
  - Message management (Add System/User/Assistant messages)
  - Run Chat button

- **Chat Interface** (Main Area):
  - Editable message textareas
  - Role-based message styling
  - Delete individual messages
  - Loading states
  - Error handling
  - Auto-scroll to latest message

## 🎨 Theming

### Dark Mode
Toggle between light and dark mode using the sun/moon icon in the navbar. Theme preference is persisted in localStorage.

### Tailwind Configuration

Custom color palette defined in `tailwind.config.js`:
```javascript
colors: {
  primary: {
    50: '#f0f9ff',
    // ... full color scale
    950: '#172554'
  }
}
```

## 🔌 API Integration

### API Service (`src/services/api.js`)

Centralized API client with methods:
- `verifyApiKey(apiKey)` - Authenticate user
- `getApiKeys()` - List all keys (admin)
- `createApiKey(name, role, rateLimit)` - Create key (admin)
- `deleteApiKey(keyId)` - Delete key (admin)
- `getStats()` - Get statistics
- `chatCompletion(model, messages, options)` - Chat with AI
- `listModels()` - Get available models

All requests include:
- `Authorization: Bearer <API_KEY>` header
- Error handling with try/catch
- Consistent response format: `{ success, data, error }`

## 📊 State Management

### Context Providers

1. **AuthContext** (`src/context/AuthContext.jsx`)
   - User authentication state
   - API key storage
   - User role (admin/user)
   - Login/logout functions

2. **ThemeContext** (`src/context/ThemeContext.jsx`)
   - Dark/light mode state
   - Theme toggle function
   - LocalStorage persistence

## 🎯 Routing

Protected routes with role-based access:

| Route | Access | Component |
|-------|--------|-----------|
| `/` | Public | Home |
| `/signin` | Public | SignIn |
| `/dashboard` | User/Admin | UserDashboard |
| `/admin` | Admin Only | AdminDashboard |
| `/playground` | User/Admin | Playground |

## 🚨 Troubleshooting

### Port 3000 Already in Use
```bash
# Find and kill process
lsof -i :3000
kill -9 <PID>

# Or use different port
npm run dev -- --port 3001
```

### API Connection Issues
1. Check backend is running: `curl http://localhost:8000/health`
2. Verify `VITE_API_URL` in `.env`
3. Check browser console for CORS errors
4. Ensure backend CORS is configured for `http://localhost:3000`

### Docker Volume Issues
```bash
# Remove all containers and volumes
docker compose down -v

# Rebuild from scratch
docker compose up --build -d
```

### Hot Reload Not Working
1. Ensure you're using volume mounts in docker-compose
2. Check file watchers aren't maxed out (Linux):
   ```bash
   echo fs.inotify.max_user_watches=524288 | sudo tee -a /etc/sysctl.conf
   sudo sysctl -p
   ```

## 📦 Dependencies

Key packages (see `package.json` for complete list):

```json
{
  "react": "^18.3.1",
  "react-dom": "^18.3.1",
  "react-router-dom": "^6.28.0",
  "axios": "^1.7.9",
  "lucide-react": "^0.468.0",
  "recharts": "^2.15.0",
  "tailwindcss": "^3.4.17"
}
```

## 🔐 Security Considerations

1. **API Keys**: Never commit real API keys to version control
2. **Environment Variables**: Use `.env` files (gitignored)
3. **HTTPS**: Use HTTPS in production
4. **CSP**: Configure Content Security Policy headers
5. **XSS**: React automatically escapes content

## 🌐 Browser Support

- Chrome/Edge (latest 2 versions)
- Firefox (latest 2 versions)
- Safari (latest 2 versions)
- Mobile browsers (iOS Safari, Chrome Mobile)

## 🚀 Deployment

### Production Build

```bash
# Build optimized production bundle
npm run build

# Serve with any static file server
npx serve -s dist
```

### Environment Variables for Production

```env
VITE_API_URL=https://your-api-domain.com
```

### Docker Production Deployment

Update `docker-compose.yml` with production configuration:
- Use environment-specific `.env` files
- Configure proper domains
- Set up SSL certificates
- Use production-grade web server (nginx)

## 📚 Additional Resources

- [React Documentation](https://react.dev)
- [Vite Guide](https://vitejs.dev/guide)
- [Tailwind CSS](https://tailwindcss.com/docs)
- [React Router](https://reactrouter.com)

## 🤝 Development Workflow

1. Create feature branch
2. Make changes
3. Test locally: `npm run dev`
4. Test with Docker: `docker compose up --build`
5. Build for production: `npm run build`
6. Commit and push
7. Create pull request

## 📝 License

[Your License Here]

## 👥 Contributors

[Your Team/Contributors]

---

**Created**: December 23, 2025  
**Version**: 1.0.0
