# Gene-Disease Analysis Frontend

A modern React application with Redux Toolkit for gene-disease relationship analysis with real-time updates and comprehensive state management.

## Features

- **User Authentication**: Secure credential management with username and API key validation
- **Gene-Disease Analysis**: Interactive forms for submitting analysis requests
- **Real-time Updates**: Live progress tracking with streaming results display
- **Analysis History**: Complete history with expandable result details
- **Responsive Design**: Clean, accessible interface optimized for research workflows
- **State Management**: Redux Toolkit for predictable state updates and optimized re-renders

## Tech Stack

- **React 19**: Modern React with hooks and concurrent features
- **Redux Toolkit**: Simplified Redux with async thunks and RTK Query patterns  
- **Webpack**: Module bundling with hot reload for development
- **CSS3**: Clean styling with CSS Grid and Flexbox layouts
- **ES6+**: Modern JavaScript with async/await and destructuring

## Quick Start

### 1. Prerequisites

Make sure you have Node.js installed:

```bash
# Check Node.js version (requires Node 16+)
node --version
npm --version

# Install Node.js if needed (recommended: use nvm)
# macOS/Linux:
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install node

# Windows: Download from https://nodejs.org/
```

### 2. Installation

```bash
# Navigate to the frontend directory
cd frontend

# Install dependencies
npm install

# This installs all React, Redux, and build dependencies
```

### 3. Running the Application

```bash
# Development server with hot reload (recommended)
npm start

# Or using the underlying webpack command
npx webpack serve --mode development

# Build for production
npm run build
```

The application will be available at:
- **Frontend**: http://localhost:3000
- **Hot Reload**: Automatically reloads on file changes
- **Source Maps**: Available in development mode

### 4. Environment Configuration

The frontend connects to the backend API with configurable endpoints:

```javascript
// Default configuration (can be overridden)
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const API_ENDPOINTS = {
  session: `${API_BASE_URL}/session`,
  analysis: `${API_BASE_URL}/analysis`,
  health: `${API_BASE_URL}/health`
};
```

Optional environment variables (create `.env` file):

```bash
# API Configuration
REACT_APP_API_URL=http://localhost:8000
REACT_APP_DEBUG=true

# Development settings
GENERATE_SOURCEMAP=true
FAST_REFRESH=true
```

## Application Structure

The application follows a modular, scalable architecture:

```
frontend/
├── src/
│   ├── components/              # Reusable UI components
│   │   ├── AuthForm.js          # Authentication form with validation
│   │   ├── AnalysisForm.js      # Gene-disease analysis input form
│   │   ├── ResultsDisplay.js    # Real-time streaming results
│   │   └── AnalysisHistory.js   # Collapsible analysis history
│   ├── features/                # Redux slices and feature logic
│   │   ├── auth/
│   │   │   └── authSlice.js     # Authentication state management
│   │   └── analysis/
│   │       └── analysisSlice.js # Analysis state with async thunks
│   ├── services/                # API and external service integrations
│   │   └── mockApi.js           # Mock API service for development
│   ├── store/                   # Redux store configuration
│   │   └── index.js             # RTK store setup with middleware
│   ├── App.js                   # Main application component
│   ├── App.css                  # Application styling
│   └── index.js                 # React application entry point
├── public/
│   └── index.html               # HTML template
├── package.json                 # Dependencies and npm scripts
├── webpack.config.js            # Webpack build configuration
└── README.md                    # This file
```

## Key Components

### Authentication Flow

**AuthForm Component** (`src/components/AuthForm.js`)
- Username and API key input with real-time validation
- Regex validation for OpenAI/Anthropic API key formats
- Error handling with user-friendly messages
- Automatic state management via Redux

**Features:**
- Form validation with instant feedback
- Secure credential handling (never persisted)
- Responsive design for various screen sizes

### Analysis Workflow

**AnalysisForm Component** (`src/components/AnalysisForm.js`)
- Gene and disease name input with sanitization
- Real-time form validation and error handling
- Disabled state during active analysis
- Integration with Redux for seamless state updates

**ResultsDisplay Component** (`src/components/ResultsDisplay.js`)
- Real-time progress updates with streaming simulation
- Visual progress indicators and loading states
- Error handling with retry capabilities
- Memoized for optimal rendering performance

**AnalysisHistory Component** (`src/components/AnalysisHistory.js`)
- Reverse chronological display of all analyses
- Collapsible result details for space efficiency
- Timestamp formatting and result previews
- Optimized re-rendering with React.memo

## State Management

### Redux Toolkit Architecture

**Authentication Slice** (`src/features/auth/authSlice.js`)
```javascript
// State structure
{
  username: string,
  apiKey: string,
  isAuthenticated: boolean
}

// Actions
- setCredentials: Store user credentials
- clearCredentials: Clear authentication state
```

**Analysis Slice** (`src/features/analysis/analysisSlice.js`)
```javascript
// State structure
{
  currentAnalysis: {
    isLoading: boolean,
    gene: string,
    disease: string,
    progress: string,
    error: string | null
  },
  history: AnalysisResult[]
}

// Async Actions
- analyzeGeneDisease: Submit analysis with streaming updates
- updateAnalysisProgress: Handle real-time progress updates
```

### Optimized Performance

- **React.memo**: All components memoized to prevent unnecessary re-renders
- **Selective Selectors**: Redux selectors minimize component subscriptions
- **Async Thunks**: Non-blocking API calls with proper error handling
- **Code Splitting**: Webpack configuration supports dynamic imports

## API Integration

### Backend Communication

The frontend integrates seamlessly with the FastAPI backend:

**Session Management:**
```javascript
// Create session
POST /session
{
  "username": "researcher01",
  "api_key": "sk-..."
}
```

**Analysis Workflow:**
```javascript
// Start analysis
POST /analysis
{
  "gene": "BRCA1",
  "disease": "breast cancer", 
  "session_id": "uuid"
}

// Get results
GET /analysis/{analysis_id}

// Stream progress (mock simulation)
Real-time updates via Redux thunks
```

**Error Handling:**
- Network error recovery with user feedback
- Validation error display with field highlighting
- Session expiration handling with re-authentication prompts

## Development Workflow

### Available Scripts

```bash
# Development
npm start              # Start development server with hot reload
npm run build          # Create production build
npm test               # Run test suite (when implemented)

# Linting and Formatting (future enhancement)
npm run lint           # ESLint code analysis
npm run format         # Prettier code formatting
npm run type-check     # TypeScript checking (if migrated)
```

### Development Best Practices

**Code Organization:**
- Feature-based folder structure for scalability
- Component separation: presentational vs container components
- Redux slices co-located with related logic
- Consistent naming conventions throughout

**Performance Optimization:**
- Memoization strategy for expensive computations
- Efficient Redux selector usage
- Webpack optimization for bundle size
- CSS optimization for fast loading

**Code Quality:**
- Consistent ES6+ usage with modern JavaScript patterns
- Comprehensive JSDoc comments explaining component purposes
- Error boundaries for graceful error handling (future enhancement)
- Accessibility considerations in component design

## Usage Examples

### Basic Workflow

1. **Authentication**
```javascript
// User enters credentials
username: "researcher01"
apiKey: "sk-abcd1234efgh5678..."

// Redux automatically validates and stores
dispatch(setCredentials({ username, apiKey }));
```

2. **Analysis Submission**
```javascript
// User submits analysis
gene: "BRCA1"
disease: "breast cancer"

// Async thunk handles API communication
dispatch(analyzeGeneDisease({ gene, disease, apiKey }));
```

3. **Real-time Updates**
```javascript
// Mock streaming updates displayed in real-time
"Starting analysis pipeline..."
"Searching databases for gene: BRCA1..."
"Analysis completed!"
```

4. **History Management**
```javascript
// Previous analyses automatically saved and displayed
history: [
  {
    id: "uuid",
    gene: "BRCA1", 
    disease: "breast cancer",
    result: "Comprehensive analysis results...",
    timestamp: "2025-01-15T10:30:00Z"
  }
]
```

## Production Deployment

### Build Process

```bash
# Create optimized production build
npm run build

# Output directory: frontend/dist/
# Contains minified JavaScript, CSS, and assets
# Ready for deployment to any static hosting service
```

### Deployment Options

**Static Hosting:**
```bash
# Netlify
netlify deploy --dir=dist

# Vercel  
vercel --prod

# AWS S3 + CloudFront
aws s3 sync dist/ s3://your-bucket-name
```

**Docker Deployment:**
```dockerfile
# Multi-stage Docker build
FROM node:18-alpine as builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 80
```

**Environment Variables for Production:**
```bash
REACT_APP_API_URL=https://api.yourdomain.com
REACT_APP_DEBUG=false
GENERATE_SOURCEMAP=false
```

## Integration with Backend

### CORS Configuration

The frontend is configured to work with the FastAPI backend's CORS settings:

```javascript
// Backend allows these origins by default:
- http://localhost:3000  (React development)
- http://localhost:3001  (Alternative port)
- http://127.0.0.1:3000  (IP format)
```

### API Compatibility

- **Authentication**: Compatible with backend session management
- **Real-time Updates**: Integrates with Server-Sent Events (SSE) for live progress
- **Error Handling**: Matches backend error response formats
- **Data Validation**: Client-side validation mirrors backend Pydantic models

## Future Enhancements

### Development Roadmap

**Testing Implementation:**
- Unit tests with Jest and React Testing Library
- Integration tests for Redux workflows
- End-to-end tests with Cypress

**Performance Optimization:**
- React.lazy for code splitting
- Service Worker for offline functionality
- Redux Persist for state persistence
- Virtual scrolling for large history lists

**User Experience:**
- Dark mode theme support
- Keyboard navigation and accessibility improvements
- Progressive Web App (PWA) capabilities
- Advanced filtering and search for analysis history

**TypeScript Migration:**
- Gradual migration to TypeScript for better type safety
- Enhanced IDE support and developer experience
- Improved error catching during development

## Troubleshooting

### Common Issues

**Development Server Won't Start:**
```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Check Node.js version compatibility
node --version  # Should be 16+
```

**Backend Connection Issues:**
```bash
# Verify backend is running
curl http://localhost:8000/health

# Check CORS configuration
# Ensure frontend origin is in backend's ALLOWED_ORIGINS

# Verify API endpoints
# Check network tab in browser dev tools
```

**State Management Issues:**
```bash
# Use Redux DevTools Extension for debugging
# Available at: https://redux-devtools-extension.js.org/

# Check Redux store state
# Use React Developer Tools for component debugging
```

**Build Issues:**
```bash
# Clear webpack cache
rm -rf node_modules/.cache

# Check for conflicting dependencies
npm ls

# Verify webpack configuration
npx webpack --mode=development --dry-run
```

## Contributing

### Development Setup

1. Fork and clone the repository
2. Install dependencies: `npm install`
3. Start development server: `npm start`
4. Make changes and test thoroughly
5. Follow existing code style and patterns
6. Submit pull request with clear description

### Code Style Guidelines

- Use functional components with hooks
- Implement React.memo for performance optimization
- Follow Redux Toolkit patterns for state management
- Write descriptive comments explaining component purposes
- Use consistent naming conventions (camelCase for variables, PascalCase for components)
- Keep components focused and single-purpose

## License
Developed by Shitlesh Bakshi on August 2025. 
