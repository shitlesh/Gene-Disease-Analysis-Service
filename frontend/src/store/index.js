import { configureStore } from '@reduxjs/toolkit';
import authSlice from '../features/auth/authSlice';
import analysisSlice from '../features/analysis/analysisSlice';

/**
 * Redux store configuration using RTK
 * Combines auth and analysis feature slices
 */
export const store = configureStore({
  reducer: {
    auth: authSlice,
    analysis: analysisSlice,
  },
  // Enable Redux DevTools in development
  devTools: process.env.NODE_ENV !== 'production',
});

// Export types for TypeScript support (optional, but good practice)
export const RootState = store.getState;
export const AppDispatch = store.dispatch;