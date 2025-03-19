import ts from '@typescript-eslint/eslint-plugin'
import tsParser from '@typescript-eslint/parser'
import prettierConfig from 'eslint-config-prettier'

export default [
	{
		languageOptions: {
			parser: tsParser,
			ecmaVersion: 'latest',
			sourceType: 'module',
		},
		plugins: {
			'@typescript-eslint': ts,
		},
		rules: {
			...ts.configs.recommended.rules, // Load recommended TypeScript rules
			...prettierConfig.rules,
			'brace-style': ['error', '1tbs'], // Enforce 1tbs brace style
			curly: ['error', 'all'], // Enforce curly braces
			indent: ['error', 'tab'], // Enforce tabs
			'max-lines-per-function': 'off', // Disable max lines per function
			'nonblock-statement-body-position': ['error', 'below'],
			'@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_', varsIgnorePattern: '^_' }],
		},
	},
]
