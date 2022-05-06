# Fig. 20.1: fig20_01.py
# This program displays the contents of a file on a Web server.
from Tkinter import *
import Pmw
import urllib
import urlparse

class WebBrowser( Frame ):
 "A simple Web browser"

	def __init__( self ):
		"Create the Web browser GUI"

		Frame.__init__( self )
		Pmw.initialise()
		self.pack( expand = YES, fill = BOTH )
		self.master.title( "Simple Web Browser" )
		self.master.geometry( "400x300" )

		self.address = Entry( self )
		self.address.pack( fill = X, padx = 5, pady = 5 )
		self.address.bind( "<Return>", self.getPage )

		self.contents = Pmw.ScrolledText( self,
		text_state = DISABLED )
		self.contents.pack( expand = YES, fill = BOTH, padx = 5,
		pady = 5 )

	def getPage( self, event ):
		 "Parse the URL, add addressing scheme and retrieve file"

		 # parse the URL
		 myURL = event.widget.get()
		 components = urlparse.urlparse( myURL )
		 self.contents.text_state = NORMAL

		 # if addressing scheme not specified, use http
		 if components[ 0 ] == "":
		    myURL = "http://" + myURL

		 # connect and retrieve the file
		 try:
			tempFile = urllib.urlopen( myURL )
			self.contents.settext( tempFile.read() ) # show results
		 except IOError:
		    self.contents.settext( "Error finding file" )

		 self.contents.text_state = DISABLED

def main():
 	WebBrowser().mainloop()

if __name__ == "__main__":
	main()