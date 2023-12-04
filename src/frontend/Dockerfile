# # Stage 1: Build the React application
# FROM node:16.0-alpine as build

# WORKDIR /app
# ENV PATH /app/node_modules/.bin:$PATH
# ENV PUBLIC_URL /

# COPY package.json ./
# COPY yarn.lock ./
# COPY vite.config.js ./
# RUN yarn install

# COPY . ./
# RUN yarn run build

# Stage 2: Serve the static files using Nginx
FROM nginx:stable
# COPY --from=build /app/build /usr/share/nginx/html
COPY dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]