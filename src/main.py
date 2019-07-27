import discord
import os
import fortune
import boi



class ELLie(discord.Client):

	FORTUNE_TELLER = fortune.FortuneTeller('res/fortunes')

	@staticmethod
	def handle_command(command):
		tokens = command.split(' ')

		if tokens[0] == 'fortune':
			fortune = ELLie.FORTUNE_TELLER.tell()
			return f"'\n`{fortune}`"
		elif command.startswith('```'):
			command.replace('`', '')
			return boi.run(command.replace('`', ''))
		else:
			return "No such command."

	async def on_message(self, message):
		if message.content == 'ELL':
			for ch in ["📠", "🇫", "🇷", "🇴", "🇬", "🇳", "🇨"]:
				await message.add_reaction(ch)
		elif message.content.startswith('%'):
			res = ELLie.handle_command(message.content[1:].lstrip().rstrip())
			await message.channel.send(res)

ellie = ELLie()
ellie.run(os.environ['DISCORD_TOKEN'])