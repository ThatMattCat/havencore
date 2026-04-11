import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [sveltekit()],
	server: {
		proxy: {
			'/api': 'http://localhost:6002',
			'/ws': {
				target: 'ws://localhost:6002',
				ws: true
			},
			'/v1': 'http://localhost:6002'
		}
	}
});
