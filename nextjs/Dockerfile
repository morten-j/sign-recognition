FROM node:lts

WORKDIR /nextjs/

COPY package.json package-lock.json ./
RUN npm install

CMD npm run dev