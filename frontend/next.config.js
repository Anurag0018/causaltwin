/** @type {import('next').NextConfig} */
// The browser talks to the same-origin /api/* path; Next rewrites it to the
// FastAPI backend server-side, so local development needs no CORS setup.
// Set BACKEND_URL (server-side) to point at a non-default backend, e.g.
// http://backend:8000 under docker-compose.
const backend = process.env.BACKEND_URL || "http://localhost:8000";

const nextConfig = {
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${backend}/:path*` }];
  },
};

module.exports = nextConfig;
